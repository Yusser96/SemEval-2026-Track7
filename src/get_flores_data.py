

from datasets import load_dataset
import os
import json




def get_flores_language_mapping():
    return {'am-ET': 'amh_Ethi',
 'ar-DZ': 'kab_Latn',
 'ar-EG': 'arz_Arab',
 'ar-MA': 'ary_Arab',
 'ar-SA': 'ars_Arab',
 'as-AS': 'asm_Beng',
 'az-AZ': 'azj_Latn',
 'bg-BG': 'bul_Cyrl',
 'el-GR': 'ell_Grek',
 'en-AU': 'eng_Latn',
 'en-AZ': 'azj_Latn',
 'en-BG': 'bul_Cyrl',
 'en-CN': 'zho_Hans',
 'en-DZ': 'kab_Latn',
 'en-EG': 'arb_Latn',
 'en-ES': 'spa_Latn',
 'en-ET': 'amh_Ethi',
 'en-FR': 'fra_Latn',
 'en-GB': 'eng_Latn',
 'en-GR': 'ell_Grek',
 'en-ID': 'ind_Latn',
 'en-IE': 'gle_Latn',
 'en-IR': 'pes_Arab',
 'en-JP': 'jpn_Jpan',
 'en-KR': 'kor_Hang',
 'en-LK': 'sin_Sinh',
 'en-MA': 'arb_Latn',
 'en-NG': 'hau_Latn',
 'en-PH': 'tgl_Latn',
 'en-SA': 'ars_Arab',
 'en-SE': 'swe_Latn',
 'en-TW': 'zho_Hant',
 'en-US': 'eng_Latn',
 'es-EC': 'spa_Latn',
 'es-ES': 'spa_Latn',
 'es-MX': 'spa_Latn',
 'eu-PV': 'eus_Latn',
 'eu-ES': 'eus_Latn',
 'fa-IR': 'pes_Arab',
 'fr-FR': 'fra_Latn',
 'ga-IE': 'gle_Latn',
 'ha-NG': 'hau_Latn',
 'id-ID': 'ind_Latn',
 'ja-JP': 'jpn_Jpan',
 'ko-KP': 'kor_Hang',
 'ko-KR': 'kor_Hang',
 'su-JB': 'jav_Latn',
 'sv-SE': 'swe_Latn',
 'ta-LK': 'sin_Sinh',
 'tl-PH': 'tgl_Latn',
 'zh-CN': 'zho_Hans',
 'zh-SG': 'zsm_Latn',
 'zh-TW': 'zho_Hant',
 "ms-SG": "zsm_Latn",
 "ta-SG": "tam_Taml",

 "en-AS":"asm_Beng",
"en-PV":"eus_Latn",
"en-KP":"kor_Hang",
"en-EC":"spa_Latn",
"en-MX":"spa_Latn",
"en-JB":"jav_Latn",
"en-SG":"tam_Taml",
 }
    

def load_flores_data(split: str = "dev", data_path: str = None):
    """Load FLORES-200 dataset for translation from HuggingFace"""
    # Load the dataset from HuggingFace

    lang_codes_mapping = get_flores_language_mapping()
    
    data = {}

    source_lang = "eng_Latn"
    source_lang_k = "en"

    # print('" "'.join(list(lang_codes_mapping.values())))
    # return ''
    for tgt_lang_k, tgt_lang in zip(lang_codes_mapping.keys(),lang_codes_mapping.values()):
        try:
            dataset = load_dataset("facebook/flores", name=f"{source_lang}-{tgt_lang}", trust_remote_code=True)

            if source_lang not in data:
                data[source_lang] = [{"prompt": item['sentence_' + source_lang]} for item in dataset[split]]

            if tgt_lang_k not in data:
                data[tgt_lang] = [{"prompt": item['sentence_' + tgt_lang]} for item in dataset[split]]
        except:
            print(f"Error with: {source_lang}-{tgt_lang}")


    os.makedirs("data",exist_ok=True)
    output_path = "data/flores200_dataset_low_res.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)





load_flores_data()


# print(set(list(get_flores_language_mapping().values())))

print(set(list(get_flores_language_mapping().keys())))