# app/extractors/gemini_extractor.py
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable
import re 
import math
import numpy as np

from app.config import GEMINI_API_KEY, GEMINI_MODEL, CONVERTED_DIR
from app.extractors.base import BaseExtractor
from app.extractors.context_agent import ContextAgent
from app.extractors.extraction_agent import ExtractionAgent
from app.extractors.color_mapping_agent import ColorMappingAgent
from app.utils.file_utils import convert_pdf_to_images
from app.utils.barcode_generator import add_barcodes_to_extraction_result, add_barcodes_to_products
from app.data.reference_data import (get_supplier_code, get_markup, get_category,SUPPLIER_MAP, COLOR_MAP, SIZE_MAP,CATEGORIES)
from app.utils.json_utils import safe_json_dump, fix_nan_in_products, sanitize_for_json
from app.utils.supplier_assignment import determine_best_supplier, assign_supplier_to_products

logger = logging.getLogger(__name__)

try:
    from app.utils.json_utils import safe_json_dump, fix_nan_in_products, sanitize_for_json
    has_json_utils = True
except ImportError:
    has_json_utils = False
    logger.warning("Módulo json_utils não encontrado, usar serialização padrão")

class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.context_agent = ContextAgent(api_key)
        self.extraction_agent = ExtractionAgent(api_key)
        self.ai_color_mapping_agent = ColorMappingAgent()

    async def analyze_context(self, document_path: str) -> str:
        # Delegar análise avançada para o agente de contexto
        context_info = await self.context_agent.analyze_document(document_path)
        
        # Formatar o contexto para uso pelo agente de extração
        context_description = self.context_agent.format_context_for_extraction(context_info)
        
        # Armazenar o contexto e informações para uso posterior
        self.current_context_info = context_info
        
        return context_description
    
    async def process_page(
        self, 
        image_path: str, 
        context: str,
        page_number: int,
        total_pages: int,
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self.extraction_agent.process_page(
            image_path, context, page_number, total_pages, previous_result
        )
    
    async def extract_document(
        self, 
        document_path: str,
        job_id: str,
        jobs_store: Dict[str, Any],
        update_progress_callback: Callable
    ) -> Dict[str, Any]:

        start_time = time.time()
        
        try:
            # Inicializar job
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "processing",
                "progress": 5.0
            }
            
            # ETAPA 1: Análise Avançada de Contexto e Layout
            is_pdf = document_path.lower().endswith('.pdf')
            
            if is_pdf:
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = 10.0
                logger.info(f"Análise Avançada de Contexto: Iniciando para o documento: {document_path}")
                
                # Obter contexto detalhado e informações de layout
                context_description = await self.analyze_context(document_path)
                context_info = self.current_context_info
                logger.info(f"Contexto avançado obtido com sucesso")
            else:
                context_description = "Documento de pedido ou nota de encomenda"
                context_info = {"document_type": "Documento de pedido", "supplier": "", "brand": ""}
            
            # ETAPA 2: Preparar imagens para processamento
            jobs_store[job_id]["model_results"]["gemini"]["progress"] = 15.0
            
            if is_pdf:
                # Converter todas as páginas para imagens
                image_paths = convert_pdf_to_images(document_path, CONVERTED_DIR)
            else:
                image_paths = [document_path]
                
            total_pages = len(image_paths)
            logger.info(f"Preparadas {total_pages} imagens para processamento")
            
            # ETAPA 3: Extrair produtos página por página
            combined_result = {"products": [], "order_info": {}}
            
            # Preencher informações do pedido com dados do contexto
            if context_info:
                combined_result["order_info"] = {
                    "supplier": context_info.get("supplier", ""),
                    "document_type": context_info.get("document_type", ""),
                    "order_number": context_info.get("reference_number", ""),
                    "date": context_info.get("date", ""),
                    "customer": context_info.get("customer", ""),
                    "brand": context_info.get("brand", ""),
                    "season": context_info.get("season", "")
                }
            
            # Distribuir progresso entre páginas (reservar 15% para contexto)
            progress_per_page = 80.0 / total_pages
            
            for page_num, img_path in enumerate(image_paths, start=1):
                is_first_page = (page_num == 1)
                
                # Atualizar progresso
                current_progress = 15.0 + (page_num - 1) * progress_per_page
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = current_progress
                logger.info(f"Processando página {page_num} de {total_pages}: {img_path}")
                
                # Processar a página atual
                page_result = await self.process_page(
                    img_path,
                    context_description,
                    page_num,
                    total_pages,
                    combined_result if not is_first_page else None
                )
                
                # Verificar se houve erro grave
                if "error" in page_result and not page_result.get("products"):
                    logger.error(f"Erro ao processar página {page_num}: {page_result['error']}")
                    if page_num == 1:
                        raise ValueError(f"Falha ao processar a primeira página: {page_result['error']}")
                    continue
                
                # Mesclar produtos
                if "products" in page_result:
                    combined_result["products"].extend(page_result.get("products", []))
                
                # Mesclar informações do pedido
                if "order_info" in page_result and page_result["order_info"]:
                    for key, value in page_result["order_info"].items():
                        if value and (key not in combined_result["order_info"] or not combined_result["order_info"].get(key)):
                            combined_result["order_info"][key] = value
                
                # Atualizar progresso
                jobs_store[job_id]["model_results"]["gemini"]["progress"] = 15.0 + page_num * progress_per_page
                
            if combined_result["products"]:
                try:
                    mapped_products = self.ai_color_mapping_agent.map_product_colors(
                        combined_result["products"]
                    )
                    combined_result["products"] = mapped_products
                    
                    # Obter relatório de mapeamento
                    mapping_report = self.ai_color_mapping_agent.get_mapping_report()
                    
                    # Adicionar relatório aos metadados
                    combined_result["_ai_color_mapping"] = mapping_report
                    
                    # Log de exemplos de mapeamentos para debug
                    stats = mapping_report['statistics']
                    if stats['mappings_details']:
                        for change in stats['mappings_details'][:3]:
                            confidence = change.get('confidence', 'unknown')
                            logger.info(f"  '{change['original_name']}' ({change['original_code']}) → '{change['mapped_name']}' ({change['mapped_code']}) [confidence: {confidence}]")
                
                except Exception as e:
                    logger.error(f"Erro no mapeamento AI de cores: {str(e)}")
                    combined_result["_ai_color_mapping"] = {"error": str(e)}
            else:
                logger.warning("Nenhum produto encontrado para mapeamento de cores")

            # ETAPA 4: Pós-processamento e Limpeza
            processed_products, determined_supplier = self._post_process_products(combined_result["products"], context_info)

            combined_result["order_info"]["supplier"] = determined_supplier
            
            if has_json_utils:
                supplier = context_info.get("supplier", "")
                supplier_code = get_supplier_code(supplier) if supplier else None
                markup = 2.73
                
                if supplier_code:
                    markup_value = get_markup(supplier_code)
                    if markup_value:
                        markup = markup_value
                
                # Corrigir produtos com valores NaN
                processed_products = fix_nan_in_products(processed_products, markup=markup)
                logger.info("Produtos sanitizados para evitar valores NaN no JSON")

            # Atualizar com produtos processados
            combined_result["products"] = processed_products
            logger.info(f"Pós-processamento: {len(combined_result['products'])} produtos únicos identificados")
            
            # Certificar-se de que cada produto tem suas referências e códigos de barras
            for product in combined_result["products"]:
                if "references" not in product or not product["references"]:
                    logger.warning(f"Produto sem referências: {product.get('material_code')} - {product.get('name')}")
                else:
                    barcodes_count = sum(1 for ref in product["references"] if "barcode" in ref)
            
            # Calcular tempo de processamento
            processing_time = time.time() - start_time
            
            # Adicionar informações de metadados
            combined_result["_metadata"] = {
                "pages_processed": total_pages,
                "context_description": context_description,
                "processing_approach": "two_agent_approach",
                "processing_time_seconds": processing_time,
                "context_info": context_info,
                "agents_used": ["ContextAgent", "ExtractionAgent", "ColorMappingAgent"]
            }
            
            # Atualizar job com resultado combinado
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "completed",
                "progress": 100.0,
                "result": combined_result,
                "processing_time": processing_time
            }
            
            results_file = os.path.join(os.path.dirname(CONVERTED_DIR), "results", f"{job_id}_gemini.json")
            if has_json_utils:
                success = safe_json_dump(combined_result, results_file)
                if success:
                    logger.info(f"Resultado salvo com sucesso em: {results_file}")
                else:
                    logger.error(f"Falha ao salvar resultado em: {results_file}")
            else:
                try:
                    def sanitize_basic(obj):
                        import math
                        if isinstance(obj, dict):
                            return {k: sanitize_basic(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_basic(item) for item in obj]
                        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                            return 0.0
                        else:
                            return obj
                    
                    # Sanitizar o resultado antes de serializar
                    sanitized_result = sanitize_basic(combined_result)
                    
                    with open(results_file, "w") as f:
                        json.dump(sanitized_result, f, indent=2)
                    logger.info(f"Resultado salvo com sanitização básica em: {results_file}")
                except Exception as e:
                    logger.error(f"Erro ao salvar resultado: {str(e)}")
            
            # Atualizar progresso geral do job
            update_progress_callback(job_id)
            
            return combined_result
                
        except Exception as e:
            error_message = f"Erro durante o processamento: {str(e)}"
            
            jobs_store[job_id]["model_results"]["gemini"] = {
                "model_name": GEMINI_MODEL,
                "status": "failed",
                "progress": 0.0,
                "error": error_message,
                "processing_time": time.time() - start_time
            }
            
            update_progress_callback(job_id)
            
            return {"error": error_message}
    
    def _post_process_products(self, products: List[Dict[str, Any]], context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        processed_products = []
        seen_material_codes = set()
        ref_counters = {}
        
        # ETAPA 1: DETERMINAR FORNECEDOR DO DOCUMENTO (APENAS UMA VEZ)
        supplier_name, supplier_code, markup = determine_best_supplier(context_info)
        original_brand = context_info.get("brand", "")

        # Log do resumo da determinação
        logger.info(f"Fornecedor determinado: '{supplier_name}' (código: {supplier_code}, markup: {markup})")
        
        # ETAPA 2: PROCESSAR PRODUTOS (SEM LÓGICA DE FORNECEDOR INDIVIDUAL)
        for product in products:
            # Verificar se produto tem código de material
            material_code = product.get("material_code")
            if not material_code:
                logger.warning(f"Produto sem código de material ignorado: {product.get('name', 'sem nome')}")
                continue
            
            # Limpeza do nome do produto
            product_name = product.get("name", "")
            pattern = r'^([A-Za-z\s]+)(?:\s+\d+.*)?$'
            match = re.match(pattern, product_name)
            
            if match:
                clean_name = match.group(1).strip()
                product["name"] = clean_name
            else:
                clean_name = re.sub(r'\d+', '', product_name).strip()
                clean_name = re.sub(r'\s+', ' ', clean_name).strip()
                product["name"] = clean_name
                
            # Verificar se tem cores válidas
            has_valid_colors = False
            if "colors" in product and isinstance(product["colors"], list):
                for color in product["colors"]:
                    if "sizes" in color and isinstance(color["sizes"], list) and len(color["sizes"]) > 0:
                        has_valid_colors = True
                        break
            
            # Se for produto válido, verificar duplicação
            if has_valid_colors:
                # NORMALIZAÇÃO DE CATEGORIA
                original_category = product.get("category", "")
                category_upper = original_category.upper() if original_category else ""
                
                # Garantir categoria consistente
                if any(term in category_upper for term in ['POLO', 'POLOSHIRT']):
                    normalized_category = "POLOS"
                elif any(term in category_upper for term in ['SWEATER', 'SWEAT', 'MALHA', 'JERSEY']):
                    normalized_category = "MALHAS"
                else:
                    # Para outras categorias, procurar correspondência em CATEGORIES
                    normalized_category = None
                    for category in CATEGORIES:
                        if category in category_upper or category_upper in category:
                            normalized_category = category
                            break
                    
                    # Se não encontrar, usar "ACESSÓRIOS" como fallback
                    if not normalized_category:
                        normalized_category = "ACESSÓRIOS"
                
                # Atualizar a categoria do produto
                product["category"] = normalized_category
                
                # Logging para debug
                if original_category != normalized_category:
                    logger.info(f"Categoria normalizada: '{original_category}' → '{normalized_category}' para produto '{product['name']}'")
                
                # Verificar se já processamos este produto (pelo código de material)
                if material_code in seen_material_codes:
                    
                    # Mesclar com produto existente
                    for existing_product in processed_products:
                        if existing_product.get("material_code") == material_code:
                            # Mesclar cores não duplicadas
                            existing_color_codes = {c.get("color_code") for c in existing_product.get("colors", [])}
                            
                            for color in product.get("colors", []):
                                color_code = color.get("color_code")
                                if color_code and color_code not in existing_color_codes:
                                    # Adicionar cor ainda não existente
                                    existing_product["colors"].append(color)
                                    existing_color_codes.add(color_code)
                            
                            # Recalcular total_price
                            subtotals = [color.get("subtotal", 0) for color in existing_product["colors"] 
                                        if color.get("subtotal") is not None]
                            existing_product["total_price"] = sum(subtotals) if subtotals else None
                            
                            break
                else:
                    # Novo produto, adicionar à lista de processados
                    seen_material_codes.add(material_code)
                    
                    # Inicializar contador para este código de material
                    if material_code not in ref_counters:
                        ref_counters[material_code] = 0
                    
                    # Adicionar campo de referências para cada cor e tamanho
                    product_references = []
                    
                    for color in product.get("colors", []):
                        color_code = color.get("color_code", "")
                        color_name = color.get("color_name", "")
                        
                        for size_info in color.get("sizes", []):
                            size = size_info.get("size", "")
                            quantity = size_info.get("quantity", 0)
                            
                            if quantity <= 0:
                                continue
                            
                            # Incrementar contador para este material
                            ref_counters[material_code] += 1
                            counter = ref_counters[material_code]
                            
                            # Criar referência completa
                            reference = f"{material_code}.{counter}"
                            
                            # Criar descrição formatada
                            description = f"{product['name']}[{color_code}/{size}]"
                            
                            # Adicionar referência à lista
                            product_references.append({
                                "reference": reference,
                                "counter": counter,
                                "color_code": color_code,
                                "color_name": color_name,
                                "size": size,
                                "quantity": quantity,
                                "description": description
                            })
                    
                    product["references"] = product_references
                    processed_products.append(product)
        
        # ETAPA 3: ATRIBUIR FORNECEDOR A TODOS OS PRODUTOS (APENAS UMA VEZ)
        processed_products = assign_supplier_to_products(processed_products, supplier_name, markup)
        
        # ETAPA 3.5: GARANTIR QUE TODOS OS CAMPOS ESTÃO CORRETOS
        for product in processed_products:
            # Preservar marca original se existir
            if original_brand and original_brand not in ["", "Marca não identificada"]:
                product["brand"] = original_brand
            
            # Forçar o fornecedor normalizado
            product["supplier"] = supplier_name
            
            # Garantir que cores têm fornecedor correto
            for color in product.get("colors", []):
                color["supplier"] = supplier_name
            
            # CRÍTICO: Garantir que referências têm fornecedor correto
            for reference in product.get("references", []):
                reference["supplier"] = supplier_name

        # ETAPA 4: FINALIZAR
        processed_products.sort(key=lambda p: p.get("material_code", ""))
        
        try:
            from app.utils.barcode_generator import add_barcodes_to_products
            processed_products = add_barcodes_to_products(processed_products)
        except ImportError:
            logger.warning("Módulo barcode_generator não encontrado, pulando geração de códigos de barras")
        
        # RETORNAR OS DADOS PARA ATUALIZAR O ORDER_INFO NO MÉTODO PRINCIPAL
        return processed_products, supplier_name