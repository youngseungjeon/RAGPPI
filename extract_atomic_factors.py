import re
import pandas as pd
from typing import List
from openai import OpenAI


def extract_atomic_factors(text: str, api_key: str, set_no: str) -> pd.DataFrame:
    client = OpenAI(api_key=api_key)

    prompt = f"""You are an expert biomedical writer. 
Please break down the following sentence into independent atomic facts. 

Examples:
Input: PIH1D1 stabilizes mTORC1, promoting cell growth and protein synthesis in breast cancer, making it a potential target for therapeutic intervention. PIH1D1 interacts with Raptor to regulate mTORC1 assembly, enhancing S6 kinase phosphorylation and rRNA transcription, while its knockdown decreases mTORC1 activity.
Output:
1. PIH1D1 stabilizes mTORC1.
2. Stabilization of mTORC1 by PIH1D1 promotes cell growth.
3. Stabilization of mTORC1 by PIH1D1 promotes protein synthesis in breast cancer.
4. PIH1D1 is a potential target for therapeutic intervention in breast cancer.
5. PIH1D1 interacts with Raptor.
6. The interaction between PIH1D1 and Raptor regulates mTORC1 assembly.
7. Regulation of mTORC1 assembly by PIH1D1 enhances S6 kinase phosphorylation.
8. Regulation of mTORC1 assembly by PIH1D1 enhances rRNA transcription.
9. Knockdown of PIH1D1 decreases mTORC1 activity.

Input: Disruption of BRCA1 interactions with BARD1 and BACH1 impairs DNA repair, increasing sensitivity to DNA damage and contributing to cancer progression. Mutations in the BRCT and RING domains of BRCA1 prevent its proper localization in DNA damage repair foci, despite maintaining some protein-protein interactions.
Output:
1. BRCA1 interacts with BARD1.
2. BRCA1 interacts with BACH1.
3. Disruption of BRCA1-BARD1 and BRCA1-BACH1 interactions impairs DNA repair.
4. Impaired DNA repair increases sensitivity to DNA damage.
5. Increased sensitivity to DNA damage contributes to cancer progression.
6. Mutations in the RING domain of BRCA1 prevent its proper localization to DNA repair foci.
7. Mutated BRCA1 retains some protein-protein interactions despite mislocalization.

Input: Targeting NEMO’s interaction with Lys 63-linked polyubiquitin or RIP could inhibit NF-κB activation, which may be beneficial in treating inflammatory diseases and certain cancers. NEMO binds to Lys 63-linked polyubiquitin to recruit IKK to TNF receptor 1, leading to NF-κB activation. Mutations preventing this binding disrupt IKK activation and NF-κB signaling.
Output:
1. NEMO binds to Lys 63-linked polyubiquitin.
2. NEMO binds to RIP.
3. NEMO binding to Lys 63-linked polyubiquitin recruits IKK to TNF receptor 1.
4. Recruitment of IKK to TNF receptor 1 leads to NF-κB activation.
5. Targeting NEMO’s interaction with Lys 63-linked polyubiquitin could inhibit NF-κB activation.
6. Targeting NEMO’s interaction with RIP could inhibit NF-κB activation.
7. Inhibiting NF-κB activation may be beneficial for treating inflammatory diseases.
8. Inhibiting NF-κB activation may be beneficial for treating certain cancers.
9. Mutations that prevent NEMO binding to Lys 63-linked polyubiquitin disrupt IKK activation.
10. Disruption of IKK activation impairs NF-κB signaling.


Now extract atomic facts from:
\"\"\"{text}\"\"\"

Atomic Facts:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert biomedical writer. Extract atomic facts from biomedical text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    facts = []
    for line in content.split("\n"):
        match = re.match(r"^\s*\d+\.\s*(.+)$", line.strip())
        if match:
            facts.append(match.group(1).strip())

    # 🛡️ fallback: 마침표 기준 분리
    if not facts:
        text_str = str(text)  # 안전하게 str 변환
        sentences = re.split(r"[.!?]+\s+", text_str)
        facts = [s.strip() for s in sentences if s.strip()]

    # ✅ DataFrame으로 반환 (Set_No / Atomic_facts)
    df_facts = pd.DataFrame({"Set_No": [set_no] * len(facts), "Atomic_facts": facts})

    return df_facts
