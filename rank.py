import argparse
import sys
import re
import json
import csv
import random
from datetime import datetime
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util


SKILL_WEIGHTS = {
    "learning to rank": 6,
    "ranking systems": 6,
    "retrieval systems": 6,
    "semantic search": 5,
    "vector databases": 4,
    "embeddings": 4,
    "production ml systems": 5,
    "python": 3,
    "ml systems": 3,
}

JD_PHRASES = list(SKILL_WEIGHTS.keys())

PHRASE_SYNONYMS = {
    "vector databases": ["pinecone", "weaviate", "milvus", "qdrant", "faiss", "opensearch", "elasticsearch"],
    "embeddings": ["sentence transformers", "text encoders", "vector representations", "bge", "e5"],
    "retrieval systems": ["information retrieval", "search systems", "search backend", "hybrid search"],
    "ranking systems": ["learning to rank", "recommendation systems", "ndcg", "mrr", "map"],
    "production ml systems": ["mlops", "docker", "kubernetes", "ab testing", "drift"]
}

STRONG_VERBS = {"built", "deployed", "shipped", "owned", "scaled", "production", "migrated", "designed", "implemented", "optimized"}
WEAK_VERBS = {"learning", "experimenting", "trying", "course", "tutorial", "certified"}
PROJECT_KEYWORDS = {"retrieval", "ranking", "semantic search", "vector search", "recommendation", "embeddings", "rag"}

CONSULTING_PATTERNS = re.compile(r'\b(tcs|infosys|wipro|accenture|cognizant|capgemini|genpact|hcl|deloitte|pwc|ey|kpmg|ibm|consulting|consultant|agency)\b', re.I)
RESEARCH_PATTERNS = re.compile(r'\b(university|institute|laboratory|research assistant|phd candidate|postdoc|published|papers|academic)\b', re.I)
MANAGEMENT_TITLES = re.compile(r'\b(manager|director|head of|vp)\b', re.I)
MANAGEMENT_VERBS = re.compile(r'\b(managed|directed|roadmap|sprints|1-on-1s|agile|leadership|stakeholders|budget)\b', re.I)
CODING_VERBS = re.compile(r'\b(built|deployed|shipped|implemented|coded|programmed|developed|optimized|scaled)\b', re.I)

MANDATORY = {"python", "embeddings", "retrieval systems", "ranking systems"}


TIER_1_CITIES = {"noida", "pune", "delhi ncr", "gurgaon", "mumbai", "hyderabad", "bangalore", "kolkata"}
PREFERRED_HUBS = {"noida", "pune"}


TARGET_PRODUCT_STARTUPS = {
    "zomato", "swiggy", "razorpay", "blinkit", "zepto", "ola", "uber", "flipkart", 
    "meesho", "phonepe", "cred", "paytm", "groww", "zerodha", "inmobi", "freshworks"
}

COMMON_BOILERPLATE_PHRASES = [
    "my academic background is in cs/ml but my main learning has come from shipping",
    "too many teams ship models without offline benchmarks they trust",
    "comfortable across the ml stack from feature engineering through deployment",
    "i've learned that most retrieval problems are actually evaluation problems in disguise"
]


def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z0-9\-\+\.]+\b", text.lower())

def normalize(text: str) -> str:
    return text.lower().strip()

def verified(term, text):
    return re.search(rf"\b{re.escape(term)}\b", text) is not None

def generate_text_blob(candidate):
    profile = candidate.get("profile", {})
    text = profile.get("summary", "") + " " + profile.get("headline", "") + " "
    for c in candidate.get("career_history", []):
        text += c.get("description", "") + " "
    return normalize(text)

def extract_skills(candidate):
    return [normalize(s.get("name", "")) for s in candidate.get("skills", []) if isinstance(s, dict)]

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def calculate_location_multiplier(candidate) -> float:
    loc = normalize(candidate.get("profile", {}).get("location", ""))
    signals = candidate.get("redrob_signals", {})
    willing_relocate = signals.get("willing_to_relocate", False)

    if any(hub in loc for hub in PREFERRED_HUBS):
        return 1.15  
    if any(city in loc for city in TIER_1_CITIES):
        return 1.0 if willing_relocate else 0.75   
    return 0.85 if willing_relocate else 0.45      

def calculate_originality_multiplier(text_blob: str) -> float:
    penalty = 0.0
    for phrase in COMMON_BOILERPLATE_PHRASES:
        if phrase in text_blob:
            penalty += 0.15  
    return max(0.4, 1.0 - penalty)

def calculate_behavioral_multiplier(candidate) -> float:
    signals = candidate.get("redrob_signals", {})
    total_penalty = 0.0
    
    last_active = signals.get("last_active_date", "")
    try:
        if last_active:
            ref_date = datetime(2026, 7, 1)
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            months_since_login = (ref_date - active_date).days / 30.4
        else:
            months_since_login = 12
    except ValueError:
        months_since_login = 0

    response_rate = signals.get("recruiter_response_rate", 1.0)
    is_open = signals.get("open_to_work_flag", True)
    interview_rate = signals.get("interview_completion_rate", 1.0)

    if months_since_login > 3:
        total_penalty += min(0.60, (months_since_login - 3) * 0.15)
    if response_rate < 0.60:
        total_penalty += max(0.0, (0.60 - response_rate) * 1.0)
    if not is_open:
        total_penalty += 0.40
    if interview_rate < 0.70:
        total_penalty += max(0.0, (0.70 - interview_rate) * 0.8)
        
    return max(0.1, 1.0 - total_penalty)

def skill_score_verified(skills, text_blob):
    skill_set = set(skills)
    score = 0
    matches = []
    mandatory_hits = 0

    for term in JD_PHRASES:
        weight = SKILL_WEIGHTS.get(term, 3)
        if term in skill_set:
            if verified(term, text_blob):
                score += weight
                matches.append(term)
                if term in MANDATORY:
                    mandatory_hits += 1
            else:
                score += weight * 0.1  

        if term in PHRASE_SYNONYMS:
            for syn in PHRASE_SYNONYMS[term]:
                if syn in skill_set:
                    if verified(syn, text_blob):
                        score += weight * 0.8
                        matches.append(syn)
                    else:
                        score += weight * 0.1

    score += (mandatory_hits * 4)
    return score, matches

def calculate_project_score(history):
    score = 0
    for job in history:
        text = normalize(job.get("description", ""))
        for v in STRONG_VERBS:
            if v in text: score += 1.5
        for v in WEAK_VERBS:
            if v in text: score -= 0.5  
        for kw in PROJECT_KEYWORDS:
            if kw in text: score += 0.8
    return np.clip(score, 0, 10)

def role_bonus(candidate):
    title = normalize(candidate.get("profile", {}).get("current_title", ""))
    if any(x in title for x in ["writer", "sales", "marketing", "hr", "civil", "mechanical"]):
        return -25
    if any(x in title for x in ["ml", "ai", "data scientist", "search", "retrieval", "ranking"]):
        return 12
    return 0

def step1_pipeline(candidates, jd_text, top_k=800):
    print(f"Stage 1: Processing Lexical Parsing pipeline across {len(candidates)} records...")
    valid_candidates = []
    docs = []
    
    for idx, c in enumerate(candidates):
        valid_candidates.append(c)
        docs.append(tokenize(generate_text_blob(c)))

    if not docs:
        return []

    bm25 = BM25Okapi(docs)
    jd_tokens = tokenize(jd_text)
    bm25_raw = bm25.get_scores(jd_tokens)
    max_bm25 = max(bm25_raw) if len(bm25_raw) else 1

    results = []
    for idx, cand in enumerate(valid_candidates):
        text_blob = generate_text_blob(cand)
        skills = extract_skills(cand)

        skill_score, matches = skill_score_verified(skills, text_blob)
        proj_score = calculate_project_score(cand.get("career_history", []))
        role_b = role_bonus(cand)
        bm25_score = (bm25_raw[idx] / max_bm25) * 30

        final_score = (bm25_score * 1.2) + skill_score + (proj_score * 2.0) + role_b
        
        final_score *= calculate_location_multiplier(cand)
        final_score *= calculate_originality_multiplier(text_blob)
        final_score *= calculate_behavioral_multiplier(cand)

        results.append({
            "candidate_data": cand,
            "candidate_id": cand.get("candidate_id", f"CAND_UNK_{idx}"),
            "score": round(final_score, 2),
            "matched_skills": matches
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def generate_candidate_chunks(candidate: dict):
    chunks = []
    profile = candidate.get("profile", {})

    identity_parts = []
    if profile.get("current_title"): identity_parts.append(f"Title: {profile.get('current_title')}")
    if profile.get("headline"): identity_parts.append(profile.get("headline"))
    
    skills = [s.get("name", "") for s in candidate.get("skills", []) if isinstance(s, dict) and s.get("name")]
    if skills: identity_parts.append("Skills: " + ", ".join(skills[:15]))
    if identity_parts: chunks.append(" ".join(identity_parts))

    for role in candidate.get("career_history", [])[:2]:
        role_parts = []
        if role.get("title"): role_parts.append(f"Role: {role.get('title')}")
        if role.get("company"): role_parts.append(f"Company: {role.get('company')}")
        if role.get("description"): role_parts.append(role.get("description"))
        if role_parts: chunks.append(" ".join(role_parts))

    if not chunks: chunks.append("Unknown background")
    return chunks

def step2_semantic_reranking(step1_results, job_query_or_emb, embedder):
    print(f"Stage 2: Initiating Bi-Encoder dense vector alignment on {len(step1_results)} items...")
    if not step1_results:
        return []

    if isinstance(job_query_or_emb, str):
        query_embedding = embedder.encode(job_query_or_emb, convert_to_tensor=True, device="cpu")
    else:
        query_embedding = job_query_or_emb

    all_chunks = []
    candidate_chunk_counts = []
    reordered_candidates = [item["candidate_data"] for item in step1_results]

    for candidate in reordered_candidates:
        chunks = generate_candidate_chunks(candidate)
        all_chunks.extend(chunks)
        candidate_chunk_counts.append(len(chunks))

    all_chunk_embeddings = embedder.encode(all_chunks, batch_size=256, convert_to_tensor=True, device="cpu", show_progress_bar=False)
    all_cos_scores = util.cos_sim(query_embedding, all_chunk_embeddings)[0]

    s1_scores = np.array([item["score"] for item in step1_results])
    s1_mean = np.mean(s1_scores) if len(s1_scores) else 0
    s1_std = np.std(s1_scores) if len(s1_scores) else 1
    if s1_std == 0: s1_std = 1

    hybrid_results = []
    current_idx = 0
    weight_template = torch.tensor([1.0, 0.9, 0.7], device="cpu")

    for i, candidate in enumerate(reordered_candidates):
        cand_id = step1_results[i]["candidate_id"]
        num_chunks = candidate_chunk_counts[i]

        candidate_scores = all_cos_scores[current_idx : current_idx + num_chunks]
        current_idx += num_chunks

        if num_chunks <= 3:
            weights = weight_template[:num_chunks]
        else:
            extra_padding = torch.full((num_chunks - 3,), 0.5, device="cpu")
            weights = torch.cat([weight_template, extra_padding])

        weighted_scores = candidate_scores * weights
        best_weighted_semantic_score = torch.max(weighted_scores).item()

        step1_score = step1_results[i]["score"]
        norm_step1 = sigmoid((step1_score - s1_mean) / s1_std)
        
        final_hybrid_score = (norm_step1 * 0.40) + (best_weighted_semantic_score * 0.60)

        hybrid_results.append({
            "candidate_data": candidate,
            "candidate_id": cand_id,
            "step1_score": step1_score,
            "matched_skills": step1_results[i]["matched_skills"],
            "semantic_score": round(best_weighted_semantic_score, 4),
            "hybrid_score": round(final_hybrid_score, 4)
        })

    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid_results


def step3_cross_encoder_rerank_with_guardrails(hybrid_candidates, ce_query, ce_model, top_k=250):
    print(f"Stage 3: Running Deep Cross-Encoder + Business Logic Guardrails on top {min(len(hybrid_candidates), top_k)} profiles...")
    pool = hybrid_candidates[:top_k]
    if not pool:
        return []
        
    pairs = []
    for item in pool:
        candidate = item["candidate_data"]
        history_text = " ".join([
            f"{role.get('title', '')} at {role.get('company', '')}: {role.get('description', '')}" 
            for role in candidate.get('career_history', [])[:2]
        ])
        skills = ", ".join([s.get('name', '') for s in candidate.get('skills', []) if isinstance(s, dict)])
        cand_blob = f"Skills: {skills}. Experience: {history_text}"
        pairs.append([ce_query, cand_blob])
        
    ce_scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
    
    ce_mean = np.mean(ce_scores) if len(ce_scores) else 0
    ce_std = np.std(ce_scores) if len(ce_scores) else 1
    if ce_std == 0: ce_std = 1
    
    for i, item in enumerate(pool):
        candidate = item["candidate_data"]
        
        norm_ce = sigmoid((ce_scores[i] - ce_mean) / ce_std)
        item["ce_score"] = round(float(norm_ce), 4)
        
        fused_score = item["hybrid_score"] * (1.0 + (norm_ce * 0.8))
        
        history_list = candidate.get('career_history', [])
        full_text_blob = f"{candidate.get('profile', {}).get('headline', '')} {candidate.get('profile', {}).get('summary', '')} " + " ".join([role.get('description', '') for role in history_list]).lower()
        current_title = normalize(candidate.get("profile", {}).get("current_title", ""))
        
        # Guardrails
        total_roles = len(history_list)
        if total_roles > 0:
            consulting_roles = sum(1 for role in history_list if CONSULTING_PATTERNS.search(role.get('company', '')))
            consulting_ratio = consulting_roles / total_roles
            current_company = history_list[0].get('company', '')
            is_current_consultant = CONSULTING_PATTERNS.search(current_company)
            
            if is_current_consultant:
                fused_score *= 0.15 
                item["ce_score"] = "PENALIZED (Active Consultant)"
            elif consulting_ratio >= 0.60:
                fused_score *= 0.40 
                item["ce_score"] = "PENALIZED (Consulting Density High)"
                
        if RESEARCH_PATTERNS.search(full_text_blob) and not CODING_VERBS.search(full_text_blob):
            fused_score *= 0.20 
            item["ce_score"] = "PENALIZED (Pure Academic)"
            
        is_manager_title = MANAGEMENT_TITLES.search(current_title)
        has_management_verbs = MANAGEMENT_VERBS.search(full_text_blob)
        has_coding_verbs = CODING_VERBS.search(full_text_blob)
        
        if (is_manager_title or has_management_verbs) and not has_coding_verbs:
            fused_score *= 0.25 
            item["ce_score"] = "PENALIZED (Non-Coding Management)"
            
        item["final_fused_score"] = float(fused_score)
        
    pool.sort(key=lambda x: x["final_fused_score"], reverse=True)
    return pool


TEMPLATES = {
    "penalized": [
        "Profile indicates misalignment with current role requirements. Assessment score adjusted based on core constraint criteria.",
        "Experience structure diverges from target technical requirements. Score reflects role-specific screening constraints.",
        "Candidate background does not fully align with the hands-on engineering focus required for this role.",
        "Technical assessment adjusted due to structural mismatch with the specific execution requirements of the position.",
        "Screening criteria indicates a deviation from the target profile; overall alignment score adjusted accordingly."
    ],
    "target_startup": [
        "Strong product background from {target_name} with {exp} years of relevant experience and demonstrated technical execution{skills_str}.",
        "Highly relevant experience from {target_name} demonstrating strong alignment with product engineering requirements ({exp} YOE){skills_str}.",
        "Top talent profile featuring {exp} years of experience at {target_name} with verified core technical competencies{skills_str}.",
        "Excellent background in high-growth environments ({target_name}), bringing {exp} years of targeted engineering experience{skills_str}.",
        "Standout candidate with {exp} years of impactful tenure at {target_name} and strong technical validation{skills_str}."
    ],
    "high_score": [
        "High alignment with core technical competencies in search, ranking, and dense vector frameworks ({exp} YOE).",
        "Exceptional technical match demonstrating deep expertise across required AI and search domains ({exp} YOE).",
        "Strong technical evaluation results highlighting advanced proficiency in core ML and search systems ({exp} YOE).",
        "Premium engineering profile with proven capabilities in targeted deep-learning and ranking frameworks ({exp} YOE).",
        "Highly qualified candidate showing comprehensive mastery of required search and vector matching technologies ({exp} YOE)."
    ],
    "standard": [
        "Qualified backend search engineering profile with {exp} years of experience in production environments.",
        "Solid technical background with {exp} years of applicable experience in backend and search engineering.",
        "Demonstrates foundational alignment with the role's backend search and production requirements ({exp} YOE).",
        "Competent engineering profile bringing {exp} years of relevant production-level experience.",
        "Meets core qualifications with {exp} years of practical experience in search-focused engineering environments."
    ],
    "demerit_location": [
        "Location falls outside primary hiring hubs, which may require remote work considerations.",
        "Current geography is outside target markets, potentially impacting hybrid collaboration.",
        "Candidate location does not align with preferred office hubs.",
        "Geographic location may necessitate flexibility regarding standard onsite policies.",
        "Based outside of core target regions for this role."
    ],
    "demerit_relocation": [
        "Candidate has indicated constraints regarding relocation to primary office hubs.",
        "Relocation preferences do not currently align with target office locations.",
        "Profile indicates an unwillingness to relocate to preferred hiring zones.",
        "Geographic constraints noted; candidate is not open to relocation.",
        "Relocation flag is active, requiring review against current remote work policies."
    ],
    "demerit_response": [
        "Recent engagement metrics suggest a passive job-seeking status.",
        "Low response rates indicate the candidate may require extended outreach efforts.",
        "Communication history suggests limited active engagement on the platform.",
        "Passive engagement indicators suggest lower immediate responsiveness.",
        "Candidate responsiveness metrics fall below standard active benchmarks."
    ],
    "demerit_interview": [
        "Historical assessment completion rates indicate potential pipeline drop-off risk.",
        "Past interview completion metrics suggest a higher risk of process attrition.",
        "Platform history shows lower-than-average follow-through on technical assessments.",
        "Engagement data indicates potential challenges with interview stage completion.",
        "Historical process completion rates require careful candidate management."
    ],
    "demerit_passive": [
        "Currently marked as not actively looking; requires a targeted sourcing approach.",
        "Profile is set to passive; candidate will need specialized outreach.",
        "Not actively on the market; requires strategic headhunting to engage.",
        "Passive talent indicator active; standard recruiting pipelines may yield lower conversion.",
        "Candidate is currently employed and not openly seeking new opportunities."
    ]
}

def build_reasoning_string(item) -> str:
    cand = item["candidate_data"]
    cand_id = item.get("candidate_id", "default")
    
    rng = random.Random(cand_id)
    
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    history = cand.get("career_history", [])
    
    exp = profile.get("years_of_experience", 0)
    has_startup_pedigree = False
    past_companies = []
    
    for job in history:
        comp = normalize(job.get("company", ""))
        past_companies.append(job.get("company", ""))
        if any(startup in comp for startup in TARGET_PRODUCT_STARTUPS):
            has_startup_pedigree = True

    matched = item.get("matched_skills", [])
    skills_str = f" ({', '.join(matched[:2])})" if matched else ""
    
    if "PENALIZED" in str(item.get("ce_score", "")):
        merit = rng.choice(TEMPLATES["penalized"])
    elif has_startup_pedigree:
        target_name = next((c for c in past_companies if normalize(c) in TARGET_PRODUCT_STARTUPS), "a premium startup")
        target_name = target_name.title()
        template = rng.choice(TEMPLATES["target_startup"])
        merit = template.format(target_name=target_name, exp=exp, skills_str=skills_str)
    elif item.get("final_fused_score", 0) > 0.75:
        template = rng.choice(TEMPLATES["high_score"])
        merit = template.format(exp=exp)
    else:
        template = rng.choice(TEMPLATES["standard"])
        merit = template.format(exp=exp)

    demerits = []
    loc = normalize(profile.get("location", ""))
    is_hub = any(hub in loc for hub in PREFERRED_HUBS)
    is_tier1 = any(city in loc for city in TIER_1_CITIES)
    willing_relocate = signals.get("willing_to_relocate", False)
    
    if not is_hub and not is_tier1:
        demerits.append(rng.choice(TEMPLATES["demerit_location"]))
    elif not is_hub and is_tier1 and not willing_relocate:
        demerits.append(rng.choice(TEMPLATES["demerit_relocation"]))
        
    resp_rate = signals.get("recruiter_response_rate", 1.0)
    interview_rate = signals.get("interview_completion_rate", 1.0)
    is_open = signals.get("open_to_work_flag", True)
    
    if resp_rate < 0.60:
        demerits.append(rng.choice(TEMPLATES["demerit_response"]))
    if interview_rate < 0.70:
        demerits.append(rng.choice(TEMPLATES["demerit_interview"]))
    if not is_open:
        demerits.append(rng.choice(TEMPLATES["demerit_passive"]))

    if demerits and "PENALIZED" not in str(item.get("ce_score", "")):
        demerit_str = " ".join(demerits[:2])
        return f"{merit} HR Notes: {demerit_str}"
        
    return merit

def execute_ranking_pipeline(candidates_list_or_path, output_csv_path="final_redrob_submission.csv"):
    if isinstance(candidates_list_or_path, str):
        raw_candidates = []
        print(f"Reading dataset directly from file: {candidates_list_or_path}")
        with open(candidates_list_or_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_candidates.append(json.loads(line))
    else:
        raw_candidates = candidates_list_or_path

    total_input_count = len(raw_candidates)
    print(f"Loaded {total_input_count} initial candidate entries.")
  
    s1_k = min(700, total_input_count)
    s3_k = min(250, s1_k)

    jd_text = "Senior AI Engineer Founding Team search ranking retrieval dense vector embeddings candidate matching system learning to rank hybrid search"
    

    s1_out = step1_pipeline(raw_candidates, jd_text, top_k=s1_k)

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    job_query = "ranker ranking retrieval search recommendation system embeddings dense hybrid vector ndcg mrr offline online evaluation A/B testing production engineering"
    s2_out = step2_semantic_reranking(s1_out, job_query, embedder)

    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    ce_query = (
        "Hands-on AI Engineer writing production Python code for a product company startup. "
        "Focuses on building search ranking models, learning to rank, dense vector retrieval, "
        "and hybrid embeddings. Prioritizes product metrics, evaluation tracking with NDCG and MRR, "
        "A/B testing, and rapid validation systems over corporate process and big-tech infrastructure."
    )
    final_sorted = step3_cross_encoder_rerank_with_guardrails(s2_out, ce_query, ce_model, top_k=s3_k)
    

    top_output_count = min(100, len(final_sorted))
    top_100 = final_sorted[:top_output_count]
    
    if top_100:
        fused_scores = np.array([item["final_fused_score"] for item in top_100])
        fused_mean = np.mean(fused_scores)
        fused_std = np.std(fused_scores) if np.std(fused_scores) != 0 else 1.0
        
        raw_normalized = [sigmoid((item["final_fused_score"] - fused_mean) / fused_std) for item in top_100]
        max_norm = max(raw_normalized)
        min_norm = min(raw_normalized)
        norm_range = max_norm - min_norm if max_norm != min_norm else 1.0
        
        for idx, item in enumerate(top_100):
            scaled_score = 0.35 + (((raw_normalized[idx] - min_norm) / norm_range) * 0.64)
            item["bounded_final_score"] = round(float(scaled_score), 4)
            
    print(f"Writing final submission file to {output_csv_path}...")
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for index, item in enumerate(top_100):
            writer.writerow([
                item["candidate_id"],
                index + 1,
                item.get("bounded_final_score", 0.0),
                build_reasoning_string(item)
            ])
            
    print(f"Pipeline completed successfully. Generated {len(top_100)} evaluated candidate rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RedRob Hackathon Candidate Ranking Pipeline")
    parser.add_argument(
        "--candidates", 
        type=str, 
        required=True, 
        help="Path to the input candidates.jsonl file"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="./submission.csv", 
        help="Path where the final submission CSV should be saved"
    )

    args = parser.parse_args()

    try:
        execute_ranking_pipeline(args.candidates, args.out)
    except FileNotFoundError:
        print(f"Error: The candidate file at '{args.candidates}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}", file=sys.stderr)
        sys.exit(1)
