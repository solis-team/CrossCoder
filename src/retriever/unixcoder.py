import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import logging

sep = "/"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_versionexec_embedding_cache = {}

_versionexec_lib_candidates_cache = {}

class UniXcoderEmbedder:
    def __init__(self, model_name="microsoft/unixcoder-base"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load UniXcoder model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def get_embedding(self, text, max_length=512):
        """Get embedding for a single text"""
        # Tokenize and encode
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state
            # Apply attention mask and mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            embeddings = torch.sum(embeddings, dim=1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()
    
    def get_batch_embeddings(self, texts, batch_size=64, max_length=512):
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = embeddings * mask_expanded
                embeddings = torch.sum(embeddings, dim=1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def compute_unixcoder_scores(query, corpus_texts, embedder):
    corpus_embeddings = embedder.get_batch_embeddings(corpus_texts, batch_size=64)
    query_embedding = embedder.get_embedding(query)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    return similarities

def retrieve_unixcoder_from_database(query, database, top_k=10, batch_size=64):
    if not database:
        return []
    
    embedder = UniXcoderEmbedder()
    
    corpus_embeddings = embedder.get_batch_embeddings(database, batch_size=batch_size)
    query_embedding = embedder.get_embedding(query)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    scored_items = []
    for i, doc in enumerate(database):
        scored_items.append({
            'doc': doc,
            'score': similarities[i]
        })
    
    sorted_items = sorted(scored_items, key=lambda x: x['score'], reverse=True)
    top_docs = [item['doc'] for item in sorted_items[:top_k]]
    
    return top_docs

def retrieve_unixcoder_from_cached_database(query, doc_indices, global_database, cached_embeddings, embedder, top_k=10):
    if not doc_indices:
        return []
    
    subset_embeddings = cached_embeddings[doc_indices]
    query_embedding = embedder.get_embedding(query)
    similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
    
    doc_texts = [global_database[idx]['doc'] for idx in doc_indices]
    
    scored_items = []
    for i, doc_text in enumerate(doc_texts):
        scored_items.append({
            'doc': doc_text,
            'score': similarities[i]
        })
    
    sorted_items = sorted(scored_items, key=lambda x: x['score'], reverse=True)
    top_docs = [item['doc'] for item in sorted_items[:top_k]]
    
    return top_docs


def retrieve_unixcoder(example, top_k=10, use_plan=False, context=None, external=True, extend=False, fine_grained=True, threshold=None, thresholds=None, is_versionexec=False):
    if thresholds is not None:
        return _retrieve_unixcoder_with_thresholds(example, thresholds=thresholds, use_plan=use_plan, context=context, external=external, extend=extend, fine_grained=fine_grained, is_versionexec=is_versionexec)
    
    if context is None:
        if fine_grained:
            context = ['class', 'function', 'method', 'variable', 'segment']
        else:
            context = ['class', 'function', 'variable']
    
    candidate = example['candidate']
    
    component_type = example.get('type')
    if component_type == 'function':
        target_prompt = example["target_function_prompt"]
    else:
        target_prompt = example["target_method_prompt"]
    
    implementation_plan = example.get('implementation_plan') if use_plan else None
    
    corpus_data = []
    candidate_texts = []
    candidate_indices = []
    
    for candidate_type in context:
        if candidate_type in candidate:
            for candidate_id, candidate_info in candidate[candidate_type].items():
                is_third_party = candidate_info.get('is_third_party')
                
                if not external and is_third_party:
                    continue
                
                candidate_source_code = candidate_info.get('source_code')
                candidate_description = candidate_info.get('description')
                candidate_text = candidate_source_code
                if candidate_description:
                    candidate_text = candidate_source_code + "\n" + candidate_description
                
                metadata = {
                    'type': candidate_type,
                    'id': candidate_id,
                    'relative_path': candidate_info['relative_path'],
                    'source_code': candidate_source_code,
                    'description': candidate_description,
                    'prompt': candidate_info.get('prompt'),
                    'is_third_party': is_third_party,
                    'library': candidate_info.get('library'),
                    'references': candidate_info.get('references')
                }
                corpus_data.append(metadata)
                candidate_texts.append(candidate_text)
                candidate_indices.append(len(corpus_data) - 1)
    
    if not corpus_data:
        return {}, {}, {}
    
    embedder = UniXcoderEmbedder()
    
    if is_versionexec:
        candidate_embeddings_list = []
        texts_to_compute = []
        indices_to_compute = []
        cache_info = []
        cached_count = 0
        total_count = len(corpus_data)
        cache_hit_libs = {}
        
        for i, (metadata, candidate_text) in enumerate(zip(corpus_data, candidate_texts)):
            library = metadata.get('library')
            candidate_id = metadata['id']
            
            cache_key = f"{candidate_id}_{candidate_text}"
            if library and library in _versionexec_embedding_cache and cache_key in _versionexec_embedding_cache[library]:
                candidate_embeddings_list.append(_versionexec_embedding_cache[library][cache_key])
                cached_count += 1
                if library:
                    cache_hit_libs[library] = cache_hit_libs.get(library, 0) + 1
            else:
                texts_to_compute.append(candidate_text)
                indices_to_compute.append(i)
                cache_info.append((library, candidate_id, cache_key) if library else (None, None, None))
                candidate_embeddings_list.append(None)
        
        task_id = example.get('id')
        if cache_hit_libs:
            hit_info = [f"{lib}({count})" for lib, count in sorted(cache_hit_libs.items())]
            logger.info(f"[UniXcoder] Task {task_id}: Cache hit {cached_count}/{total_count} candidates from libraries: {', '.join(hit_info)}")
        else:
            logger.info(f"[UniXcoder] Task {task_id}: Cache hit {cached_count}/{total_count} candidates (no cache)")
        
        if texts_to_compute:
            computed_embeddings = embedder.get_batch_embeddings(texts_to_compute, batch_size=64)
            
            cached_libs = set()
            lib_component_counts = {}
            uncached_component_count = 0
            for computed_idx, (original_idx, computed_emb) in enumerate(zip(indices_to_compute, computed_embeddings)):
                candidate_embeddings_list[original_idx] = computed_emb
                library, candidate_id, cache_key = cache_info[computed_idx]
                
                if library:
                    if library not in _versionexec_embedding_cache:
                        _versionexec_embedding_cache[library] = {}
                    if cache_key not in _versionexec_embedding_cache[library]:
                        _versionexec_embedding_cache[library][cache_key] = computed_emb
                        cached_libs.add(library)
                        lib_component_counts[library] = lib_component_counts.get(library, 0) + 1
                else:
                    uncached_component_count += 1
            
            if cached_libs:
                lib_info = [f"{lib}(+{lib_component_counts[lib]})" for lib in sorted(cached_libs)]
                if uncached_component_count > 0:
                    logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings for libraries: {', '.join(lib_info)} ({uncached_component_count} repository components not cached)")
                else:
                    logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings for libraries: {', '.join(lib_info)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings ({uncached_component_count} repository components not cached)")
            
            all_cached_libs = {lib: len(comps) for lib, comps in _versionexec_embedding_cache.items() if comps}
            if all_cached_libs:
                lib_summary = [f"{lib}({count})" for lib, count in sorted(all_cached_libs.items())]
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: {', '.join(lib_summary)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: (none)")
        else:
            all_cached_libs = {lib: len(comps) for lib, comps in _versionexec_embedding_cache.items() if comps}
            if all_cached_libs:
                lib_summary = [f"{lib}({count})" for lib, count in sorted(all_cached_libs.items())]
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: {', '.join(lib_summary)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: (none)")
        
        if len(candidate_embeddings_list) > 0:
            emb_dim = None
            for emb in candidate_embeddings_list:
                if emb is not None:
                    if emb.ndim == 1:
                        emb_dim = emb.shape[0]
                    else:
                        emb_dim = emb.shape[1]
                    break
            
            if emb_dim is None:
                candidate_embeddings = np.zeros((len(candidate_embeddings_list), 768))  
            else:
                for i, emb in enumerate(candidate_embeddings_list):
                    if emb is None:
                        candidate_embeddings_list[i] = np.zeros(emb_dim)
                candidate_embeddings = np.vstack(candidate_embeddings_list)
        else:
            candidate_embeddings = np.array([]).reshape(0, -1)
    else:
        candidate_embeddings = embedder.get_batch_embeddings(candidate_texts, batch_size=64)
    
    t_emb = embedder.get_embedding(target_prompt)[0]
    t_emb = normalize_embedding(t_emb)
    
    if len(candidate_embeddings) > 0:
        c_embs_list = [normalize_embedding(emb) if emb.ndim == 1 else normalize_embedding(emb[0]) for emb in candidate_embeddings]
    else:
        c_embs_list = []
    c_embs = np.vstack(c_embs_list) if len(c_embs_list) > 0 else np.array([]).reshape(0, -1)
    
    sim_tc = cosine_similarity(t_emb.reshape(1, -1), c_embs)[0] if len(c_embs) > 0 else np.zeros(len(candidate_texts))
    
    if use_plan and implementation_plan:
        step_embs = embedder.get_batch_embeddings(implementation_plan, batch_size=64)
        if len(step_embs) > 0:
            step_embs_list = [normalize_embedding(emb) if emb.ndim == 1 else normalize_embedding(emb[0]) for emb in step_embs]
        else:
            step_embs_list = []
        step_embs = np.vstack(step_embs_list) if len(step_embs_list) > 0 else np.array([]).reshape(0, -1)
        
        max_step_scores = np.zeros(len(candidate_texts))
        for s_i_emb in step_embs:
            q_i_raw = 0.5 * t_emb + 0.5 * s_i_emb
            q_i = normalize_embedding(q_i_raw)
            sim_qic = cosine_similarity(q_i.reshape(1, -1), c_embs)[0] if len(c_embs) > 0 else np.zeros(len(candidate_texts))
            max_step_scores = np.maximum(max_step_scores, sim_qic)
        
        final_scores = np.maximum(sim_tc, max_step_scores)
    else:
        final_scores = sim_tc
    
    scored_items = []
    all_scores = {}
    for i, metadata in enumerate(corpus_data):
        score = float(final_scores[i])
        candidate_id = metadata['id']
        candidate_type = metadata['type']
        all_scores[(candidate_type, candidate_id)] = score
        
        scored_items.append({
            'score': score,
            'metadata': metadata
        })
    
    sorted_items = sorted(scored_items, key=lambda x: x['score'], reverse=True)
    
    if threshold is not None:
        filtered_items = [item for item in sorted_items if item['score'] >= threshold]
        if use_plan and len(filtered_items) < 1:
            return {}, {}, all_scores
        top_results = filtered_items
    else:
        if use_plan:
            if len(sorted_items) < 1:
                return {}, {}, all_scores
            top_results = sorted_items[:1]
        else:
            top_results = sorted_items[:top_k] if top_k else sorted_items
    
    lib_selected = {}
    repo_selected = {}
    
    for item in top_results:
        metadata = item['metadata']
        file_path = metadata['relative_path']
        candidate_type = metadata['type']
        candidate_id = metadata['id']
        is_third_party = metadata['is_third_party']
        library = metadata['library']
        
        if is_third_party:
            if library not in lib_selected:
                lib_selected[library] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
            
            if candidate_id not in lib_selected[library][candidate_type]:
                lib_selected[library][candidate_type][candidate_id] = {
                    'relative_path': file_path,
                    'source_code': metadata.get('source_code'),
                    'prompt': metadata.get('prompt'),
                    'score': float(item['score']),
                    'library': library
                }
            else:
                if float(item['score']) > lib_selected[library][candidate_type][candidate_id]['score']:
                    lib_selected[library][candidate_type][candidate_id]['score'] = float(item['score'])
        else:
            if file_path not in repo_selected:
                repo_selected[file_path] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
            
            if candidate_id not in repo_selected[file_path][candidate_type]:
                repo_selected[file_path][candidate_type][candidate_id] = {
                    'relative_path': file_path,
                    'source_code': metadata.get('source_code'),
                    'prompt': metadata.get('prompt'),
                    'score': float(item['score']),
                    'references': metadata.get('references')
                }
            else:
                if float(item['score']) > repo_selected[file_path][candidate_type][candidate_id]['score']:
                    repo_selected[file_path][candidate_type][candidate_id]['score'] = float(item['score'])
    
    task_id = example.get('id')
    
    if extend:
        existing_component_ids = set()
        for file_path, types in repo_selected.items():
            for comp_type, comps in types.items():
                for comp_id in comps.keys():
                    existing_component_ids.add(comp_id)
        
        refs_to_add = []
        seen_ref_ids = set()
        for file_path, types in repo_selected.items():
            for comp_type, comps in types.items():
                if comp_type not in ['function', 'class', 'variable']:
                    continue
                
                for comp_id, comp_info in comps.items():
                    references = comp_info.get('references')
                    
                    if isinstance(references, list) and len(references) > 0:
                        ref_dict = references[0]
                        ref_id = ref_dict.get('id')
                        
                        if ref_id and ref_id not in seen_ref_ids:
                            ref_type = ref_dict.get('component_type')
                            
                            if ref_type in ['function', 'method']:
                                refs_to_add.append({
                                    'id': ref_id,
                                    'type': ref_type,
                                    'source_code': ref_dict.get('source_code'),
                                    'score': ref_dict.get('score')
                                })
                                seen_ref_ids.add(ref_id)
        
        added_refs = set()
        extended_component_ids = []
        for ref in refs_to_add:
            ref_id = ref['id']
            if ref_id not in added_refs and ref_id not in existing_component_ids:
                ref_type = ref['type']
                ref_file_path = ref_id.split('@')[1] if '@' in ref_id else ''
                
                if not ref_file_path:
                    continue
                
                if ref_file_path not in repo_selected:
                    repo_selected[ref_file_path] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
                
                if ref_id not in repo_selected[ref_file_path][ref_type]:
                    repo_selected[ref_file_path][ref_type][ref_id] = {
                        'relative_path': ref_file_path,
                        'source_code': ref['source_code'],
                        'prompt': ref['source_code'],
                        'score': ref['score']
                    }
                    added_refs.add(ref_id)
                    extended_component_ids.append(ref_id)
        
        if extended_component_ids:
            logger.info(f"[UniXcoder] Task {task_id}: Extended {len(extended_component_ids)} components: {extended_component_ids}")
        else:
            logger.info(f"[UniXcoder] Task {task_id}: Extend enabled but no components were extended")
    else:
        logger.info(f"[UniXcoder] Task {task_id}: Extend disabled")
    
    return lib_selected, repo_selected, all_scores

def _retrieve_unixcoder_with_thresholds(example, thresholds=None, use_plan=False, context=None, external=True, extend=False, fine_grained=True, is_versionexec=False):
    if thresholds is None:
        thresholds = []
    
    if context is None:
        if fine_grained:
            context = ['class', 'function', 'method', 'variable', 'segment']
        else:
            context = ['class', 'function', 'variable']
    
    candidate = example['candidate']
    
    component_type = example.get('type')
    if component_type == 'function':
        target_prompt = example["target_function_prompt"]
    else:
        target_prompt = example["target_method_prompt"]
    
    implementation_plan = example.get('implementation_plan') if use_plan else None
    
    corpus_data = []
    candidate_texts = []
    
    for candidate_type in context:
        if candidate_type in candidate:
            for candidate_id, candidate_info in candidate[candidate_type].items():
                is_third_party = candidate_info.get('is_third_party')
                
                if not external and is_third_party:
                    continue
                
                candidate_source_code = candidate_info.get('source_code')
                candidate_description = candidate_info.get('description')
                candidate_text = candidate_source_code
                if candidate_description:
                    candidate_text = candidate_source_code + "\n" + candidate_description
                
                metadata = {
                    'type': candidate_type,
                    'id': candidate_id,
                    'relative_path': candidate_info['relative_path'],
                    'source_code': candidate_source_code,
                    'description': candidate_description,
                    'prompt': candidate_info.get('prompt'),
                    'is_third_party': is_third_party,
                    'library': candidate_info.get('library'),
                    'references': candidate_info.get('references')
                }
                corpus_data.append(metadata)
                candidate_texts.append(candidate_text)
    
    if not corpus_data:
        return {threshold: ({}, {}) for threshold in thresholds}, {}
    
    embedder = UniXcoderEmbedder()
    
    if is_versionexec:
        candidate_embeddings_list = []
        texts_to_compute = []
        indices_to_compute = []
        cache_info = []
        cached_count = 0
        total_count = len(corpus_data)
        cache_hit_libs = {}
        
        for i, (metadata, candidate_text) in enumerate(zip(corpus_data, candidate_texts)):
            library = metadata.get('library')
            candidate_id = metadata['id']
            
            cache_key = f"{candidate_id}_{candidate_text}"
            if library and library in _versionexec_embedding_cache and cache_key in _versionexec_embedding_cache[library]:
                candidate_embeddings_list.append(_versionexec_embedding_cache[library][cache_key])
                cached_count += 1
                if library:
                    cache_hit_libs[library] = cache_hit_libs.get(library, 0) + 1
            else:
                texts_to_compute.append(candidate_text)
                indices_to_compute.append(i)
                cache_info.append((library, candidate_id, cache_key) if library else (None, None, None))
                candidate_embeddings_list.append(None)
        
        task_id = example.get('id')
        if cache_hit_libs:
            hit_info = [f"{lib}({count})" for lib, count in sorted(cache_hit_libs.items())]
            logger.info(f"[UniXcoder] Task {task_id}: Cache hit {cached_count}/{total_count} candidates from libraries: {', '.join(hit_info)}")
        else:
            logger.info(f"[UniXcoder] Task {task_id}: Cache hit {cached_count}/{total_count} candidates (no cache)")
        
        if texts_to_compute:
            computed_embeddings = embedder.get_batch_embeddings(texts_to_compute, batch_size=64)
            
            cached_libs = set()
            lib_component_counts = {}
            uncached_component_count = 0
            for computed_idx, (original_idx, computed_emb) in enumerate(zip(indices_to_compute, computed_embeddings)):
                candidate_embeddings_list[original_idx] = computed_emb
                library, candidate_id, cache_key = cache_info[computed_idx]
                
                if library:
                    if library not in _versionexec_embedding_cache:
                        _versionexec_embedding_cache[library] = {}
                    if cache_key not in _versionexec_embedding_cache[library]:
                        _versionexec_embedding_cache[library][cache_key] = computed_emb
                        cached_libs.add(library)
                        lib_component_counts[library] = lib_component_counts.get(library, 0) + 1
                else:
                    uncached_component_count += 1
            
            if cached_libs:
                lib_info = [f"{lib}(+{lib_component_counts[lib]})" for lib in sorted(cached_libs)]
                if uncached_component_count > 0:
                    logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings for libraries: {', '.join(lib_info)} ({uncached_component_count} repository components not cached)")
                else:
                    logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings for libraries: {', '.join(lib_info)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Cached {len(texts_to_compute)} new embeddings ({uncached_component_count} repository components not cached)")
            
            all_cached_libs = {lib: len(comps) for lib, comps in _versionexec_embedding_cache.items() if comps}
            if all_cached_libs:
                lib_summary = [f"{lib}({count})" for lib, count in sorted(all_cached_libs.items())]
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: {', '.join(lib_summary)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: (none)")
        else:
            all_cached_libs = {lib: len(comps) for lib, comps in _versionexec_embedding_cache.items() if comps}
            if all_cached_libs:
                lib_summary = [f"{lib}({count})" for lib, count in sorted(all_cached_libs.items())]
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: {', '.join(lib_summary)}")
            else:
                logger.info(f"[UniXcoder] Task {task_id}: Total cached libraries: (none)")
        
        if len(candidate_embeddings_list) > 0:
            # Get embedding dimension from first non-None embedding
            emb_dim = None
            for emb in candidate_embeddings_list:
                if emb is not None:
                    if emb.ndim == 1:
                        emb_dim = emb.shape[0]
                    else:
                        emb_dim = emb.shape[1]
                    break
            
            if emb_dim is None:
                # No valid embeddings, create zero embeddings
                candidate_embeddings = np.zeros((len(candidate_embeddings_list), 768))  
            else:
                # Replace None with zero embeddings
                for i, emb in enumerate(candidate_embeddings_list):
                    if emb is None:
                        candidate_embeddings_list[i] = np.zeros(emb_dim)
                candidate_embeddings = np.vstack(candidate_embeddings_list)
        else:
            candidate_embeddings = np.array([]).reshape(0, -1)
    else:
        candidate_embeddings = embedder.get_batch_embeddings(candidate_texts, batch_size=64)
    
    t_emb = embedder.get_embedding(target_prompt)[0]
    t_emb = normalize_embedding(t_emb)
    
    if len(candidate_embeddings) > 0:
        c_embs_list = [normalize_embedding(emb) if emb.ndim == 1 else normalize_embedding(emb[0]) for emb in candidate_embeddings]
    else:
        c_embs_list = []
    c_embs = np.vstack(c_embs_list) if len(c_embs_list) > 0 else np.array([]).reshape(0, -1)
    
    sim_tc = cosine_similarity(t_emb.reshape(1, -1), c_embs)[0] if len(c_embs) > 0 else np.zeros(len(candidate_texts))
    
    if use_plan and implementation_plan:
        step_embs = embedder.get_batch_embeddings(implementation_plan, batch_size=64)
        if len(step_embs) > 0:
            step_embs_list = [normalize_embedding(emb) if emb.ndim == 1 else normalize_embedding(emb[0]) for emb in step_embs]
        else:
            step_embs_list = []
        step_embs = np.vstack(step_embs_list) if len(step_embs_list) > 0 else np.array([]).reshape(0, -1)
        
        max_step_scores = np.zeros(len(candidate_texts))
        for s_i_emb in step_embs:
            q_i_raw = 0.5 * t_emb + 0.5 * s_i_emb
            q_i = normalize_embedding(q_i_raw)
            sim_qic = cosine_similarity(q_i.reshape(1, -1), c_embs)[0] if len(c_embs) > 0 else np.zeros(len(candidate_texts))
            max_step_scores = np.maximum(max_step_scores, sim_qic)
        
        final_scores = np.maximum(sim_tc, max_step_scores)
    else:
        final_scores = sim_tc
    
    scored_items = []
    all_scores = {}
    for i, metadata in enumerate(corpus_data):
        score = float(final_scores[i])
        candidate_id = metadata['id']
        candidate_type = metadata['type']
        all_scores[(candidate_type, candidate_id)] = score
        
        scored_items.append({
            'score': score,
            'metadata': metadata
        })
    
    results_by_threshold = {}
    
    for threshold in thresholds:
        lib_selected = {}
        repo_selected = {}
        
        for item in scored_items:
            score = item['score']
            if score >= threshold:
                metadata = item['metadata']
                file_path = metadata['relative_path']
                candidate_type = metadata['type']
                candidate_id = metadata['id']
                is_third_party = metadata['is_third_party']
                library = metadata['library']
                
                if is_third_party:
                    if library not in lib_selected:
                        lib_selected[library] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
                    
                    if candidate_id not in lib_selected[library][candidate_type]:
                        lib_selected[library][candidate_type][candidate_id] = {
                            'relative_path': file_path,
                            'source_code': metadata.get('source_code'),
                            'prompt': metadata.get('prompt'),
                            'score': float(score),
                            'library': library
                        }
                    else:
                        if float(score) > lib_selected[library][candidate_type][candidate_id]['score']:
                            lib_selected[library][candidate_type][candidate_id]['score'] = float(score)
                else:
                    if file_path not in repo_selected:
                        repo_selected[file_path] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
                    
                    if candidate_id not in repo_selected[file_path][candidate_type]:
                        repo_selected[file_path][candidate_type][candidate_id] = {
                            'relative_path': file_path,
                            'source_code': metadata.get('source_code'),
                            'prompt': metadata.get('prompt'),
                            'score': float(score),
                            'references': metadata.get('references')
                        }
                    else:
                        if float(score) > repo_selected[file_path][candidate_type][candidate_id]['score']:
                            repo_selected[file_path][candidate_type][candidate_id]['score'] = float(score)
        
        if extend:
            existing_component_ids = set()
            for file_path, types in repo_selected.items():
                for comp_type, comps in types.items():
                    for comp_id in comps.keys():
                        existing_component_ids.add(comp_id)
            
            refs_to_add = []
            seen_ref_ids = set()
            for file_path, types in repo_selected.items():
                for comp_type, comps in types.items():
                    if comp_type not in ['function', 'class', 'variable']:
                        continue
                    
                    for comp_id, comp_info in comps.items():
                        references = comp_info.get('references')
                        
                        if isinstance(references, list) and len(references) > 0:
                            ref_dict = references[0]
                            ref_id = ref_dict.get('id')
                            
                            if ref_id and ref_id not in seen_ref_ids:
                                ref_type = ref_dict.get('component_type')
                                
                                if ref_type in ['function', 'method']:
                                    refs_to_add.append({
                                        'id': ref_id,
                                        'type': ref_type,
                                        'source_code': ref_dict.get('source_code'),
                                        'score': ref_dict.get('score')
                                    })
                                    seen_ref_ids.add(ref_id)
            
            added_refs = set()
            for ref in refs_to_add:
                ref_id = ref['id']
                if ref_id not in added_refs and ref_id not in existing_component_ids:
                    ref_type = ref['type']
                    ref_file_path = ref_id.split('@')[1] if '@' in ref_id else ''
                    
                    if not ref_file_path:
                        continue
                    
                    if ref_file_path not in repo_selected:
                        repo_selected[ref_file_path] = {'class': {}, 'function': {}, 'method': {}, 'variable': {}, 'segment': {}}
                    
                    if ref_id not in repo_selected[ref_file_path][ref_type]:
                        repo_selected[ref_file_path][ref_type][ref_id] = {
                            'relative_path': ref_file_path,
                            'source_code': ref['source_code'],
                            'prompt': ref['source_code'],
                            'score': ref['score']
                        }
                        added_refs.add(ref_id)
        
        results_by_threshold[threshold] = (lib_selected, repo_selected)
    
    return results_by_threshold, all_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="RepoExec", help="Benchmark to run: RepoExec or DevEval")
    parser.add_argument("--imported_context", action="store_true", help="Whether to include only imported context or not")
    args = parser.parse_args()
