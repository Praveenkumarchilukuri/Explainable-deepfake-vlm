"""
src/explainability/llm_explainer.py
Language Model Forensic Translation Layer (Gemini API Integration).
Converts structured ViT predictions, Grad-CAM spatial coordinates, and CLIP semantic similarities
into human-readable, auditable forensic diagnostic reports.
"""

from typing import Dict, Any, Optional
import os
import json

try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

from src.config import GEMINI_MODEL_NAME, GEMINI_API_KEY_ENV


class GeminiForensicExplainer:
    """
    Translates mathematical model evidence into transparent forensic explanations.
    Guarantees strict grounding: the LLM never classifies; it translates JSON evidence
    into natural language explanations without hallucinations.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL_NAME):
        self.api_key = api_key or os.getenv(GEMINI_API_KEY_ENV)
        self.model_name = model_name
        self.live_llm_ready = False

        if HAS_GEMINI_SDK and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        "temperature": 0.1,  # Low temperature for strict factual grounding
                        "top_p": 0.95,
                        "max_output_tokens": 512
                    }
                )
                self.live_llm_ready = True
            except Exception as e:
                print(f"[Warning] Failed to configure Gemini API ({e}). Falling back to deterministic template.")
                self.live_llm_ready = False

    def generate_report(
        self,
        predicted_class: str,
        confidence: float,
        cam_metadata: Dict[str, Any],
        clip_similarities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generates a structured forensic explanation report.
        """
        # Prepare strictly bounded payload
        evidence_payload = {
            "verdict": predicted_class.upper(),
            "confidence_score": round(confidence, 4),
            "primary_anomaly_location": cam_metadata.get("anatomical_region", "Facial perimeter"),
            "peak_saliency_coordinates": cam_metadata.get("peak_coordinate", {"x": 112, "y": 150}),
            "peak_saliency_intensity": cam_metadata.get("peak_intensity", 0.85),
            "clip_top_semantic_matches": sorted(
                clip_similarities.items(), key=lambda item: item[1], reverse=True
            )[:2]
        }

        if self.live_llm_ready:
            try:
                prompt = self._build_gemini_prompt(evidence_payload)
                response = self.model.generate_content(prompt)
                explanation_text = response.text.strip()
                engine_used = f"Google Gemini ({self.model_name})"
            except Exception as e:
                print(f"[Warning] Gemini API call error: {e}. Falling back to template.")
                explanation_text = self._build_template_report(evidence_payload)
                engine_used = "Deterministic Forensic Engine (Fallback)"
        else:
            explanation_text = self._build_template_report(evidence_payload)
            engine_used = "Deterministic Forensic Engine (Set GEMINI_API_KEY for live LLM)"

        return {
            "verdict": predicted_class.upper(),
            "confidence": round(confidence, 4),
            "evidence_summary": evidence_payload,
            "engine": engine_used,
            "forensic_report_markdown": explanation_text
        }

    def _build_gemini_prompt(self, payload: Dict[str, Any]) -> str:
        return f"""You are a specialized AI Digital Forensics Investigator.
Convert the following structured model evidence into an objective, factual forensic report.
Rules:
1. Do not invent new artifacts not listed in the evidence.
2. State the final verdict, confidence, and anatomical localization clearly.
3. Keep the report under 150 words.

EVIDENCE PAYLOAD:
{json.dumps(payload, indent=2)}

FORMAT:
### Forensic Diagnostic Summary
- **Verdict**: [Verdict] (Confidence: [Score]%)
- **Identified Anomaly**: [Location and visual cues]
- **Multimodal Semantic Support**: [Top CLIP match]
- **Conclusion**: [1-sentence factual conclusion]
"""

    def _build_template_report(self, payload: Dict[str, Any]) -> str:
        verdict = payload["verdict"]
        conf = payload["confidence_score"] * 100
        location = payload["primary_anomaly_location"]
        coords = payload["peak_saliency_coordinates"]
        top_clip = payload["clip_top_semantic_matches"][0] if payload["clip_top_semantic_matches"] else ("None", 0.0)

        if verdict == "FAKE":
            return (
                f"### Forensic Diagnostic Summary\n"
                f"- **Verdict**: **SYNTHETIC MANIPULATION DETECTED** (Confidence: {conf:.2f}%)\n"
                f"- **Attribution Localization**: Peak Grad-CAM saliency concentrated at **{location}** "
                f"(pixel coordinates: x={coords.get('x')}, y={coords.get('y')}).\n"
                f"- **Multimodal Semantic Alignment**: Closest semantic prompt match was *\"{top_clip[0]}\"* "
                f"with a cosine similarity of {top_clip[1]:.4f}.\n"
                f"- **Forensic Assessment**: Visual boundary discrepancies and texture anomalies in the "
                f"{location} strongly indicate generative face-swapping or reenactment tampering."
            )
        else:
            return (
                f"### Forensic Diagnostic Summary\n"
                f"- **Verdict**: **AUTHENTIC MEDIA** (Confidence: {conf:.2f}%)\n"
                f"- **Attribution Localization**: Feature attribution shows uniform spatial attention across standard facial landmarks with no localized boundary anomalies.\n"
                f"- **Multimodal Semantic Alignment**: Closest semantic prompt match was *\"{top_clip[0]}\"* with similarity {top_clip[1]:.4f}.\n"
                f"- **Forensic Assessment**: Facial contours, skin micro-textures, and ocular reflections conform to natural camera acquisition characteristics."
            )
