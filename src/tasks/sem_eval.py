from datasets import load_dataset

from math import exp
from typing import Dict, List, Tuple
from typing import List, Optional
from jaxtyping import Float, Int
import torch
from tqdm import tqdm
import json
import pandas as pd

_model_output = "text"
_max_new_tokens = 64



# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")



import re


def init(device):
    import fasttext


    try:
        langid_model = fasttext.load_model('lid218e.bin')
    except:
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin', 'lid218e.bin')
        langid_model = fasttext.load_model('lid218e.bin')


    from comet import download_model, load_from_checkpoint
    # Download and load the COMET model

    model_path = download_model("Unbabel/wmt22-comet-da") #"wmt21-cometinho-da")
    comet_model = load_from_checkpoint(model_path).to(device)


    return langid_model, comet_model

def compute_comet_score(comet_model, sources, references, translations,batch_size=32,device=None):
    """
    Evaluates translations against reference translations using the COMET model.

    Args:
    sources (list of str): The source sentences.
    references (list of str): The reference translations.
    translations (list of str): The machine translations to evaluate.

    Returns:
    list of float: The COMET scores for each translation.
    """
    # Prepare the data as a list of dictionaries
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, translations, references)
    ]

    # Predict the scores
    # results = comet_model.predict(data, batch_size=batch_size,accelerator="gpu", gpus=1 if device == "cuda" else 0)
    results = comet_model.predict(
    data,
    batch_size=batch_size,
    accelerator="gpu",
    # devices=1,
)
    return results['system_score']*100


def detect_language(langid_model, text):
    """Detect language using fasttext, return language code"""
    if not text.strip():
        return "unknown"
    
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return "unknown"
    
    predictions = langid_model.predict(clean_text, k=1)
    flores_code = predictions[0][0].replace('__label__', '')
    
    # Convert FLORES code back to original simple code
    return flores_code


def compute_bleu_score(model, reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate translations"""
    #from sacrebleu.metrics.bleu import _get_tokenizer, _TOKENIZERS

    import sacrebleu
    from functools import lru_cache

    class MyTok(sacrebleu.tokenizers.tokenizer_base.BaseTokenizer):

        def signature(self):
            return 'mytok'

        @lru_cache(maxsize=2**16)
        def __call__(self, line):
            tokenizer = model.get_tokenizer()
            tokens = tokenizer.tokenize(line, add_special_tokens=False)
            return " ".join(tokens)
    

    def my_get_tokenizer(name): 
        return MyTok
    sacrebleu.metrics.bleu._get_tokenizer = my_get_tokenizer

    #from sacrebleu import sentence_bleu
    score = sacrebleu.sentence_bleu(candidate, [reference],tokenize="mytok" )
    
    return score.score




language_and_country = {
        "am-ET": {"language": "Amharic", "country": "Ethiopia"},
        "ar-DZ": {"language": "Arabic", "country": "Algeria"},
        "ar-EG": {"language": "Arabic", "country": "Egypt"},
        "ar-MA": {"language": "Arabic", "country": "Morocco"},
        "ar-SA": {"language": "Arabic", "country": "Saudi Arabia"},
        "as-AS": {"language": "Assamese", "country": "India"},
        "az-AZ": {"language": "Azerbaijani", "country": "Azerbaijan"},
        "bg-BG": {"language": "Bulgarian", "country": "Bulgaria"},
        "el-GR": {"language": "Greek", "country": "Greece"},
        "en-AU": {"language": "English", "country": "Australia"},
        "en-GB": {"language": "English", "country": "United Kingdom"},
        "en-SG": {"language": "English", "country": "Singapore"},
        "en-US": {"language": "English", "country": "United States"},
        "es-EC": {"language": "Spanish", "country": "Ecuador"},
        "es-ES": {"language": "Spanish", "country": "Spain"},
        "es-MX": {"language": "Spanish", "country": "Mexico"},
        "eu-PV": {"language": "Basque", "country": "Basque Country (Spain)"},
        "eu-ES": {"language": "Basque", "country": "Basque Country (Spain)"},
        "fa-IR": {"language": "Persian (Farsi)", "country": "Iran"},
        "fr-FR": {"language": "French", "country": "France"},
        "ga-IE": {"language": "Irish", "country": "Ireland"},
        "ha-NG": {"language": "Hausa", "country": "Nigeria"},
        "id-ID": {"language": "Indonesian", "country": "Indonesia"},
        "ja-JP": {"language": "Japanese", "country": "Japan"},
        "ko-KP": {"language": "Korean", "country": "North Korea"},
        "ko-KR": {"language": "Korean", "country": "South Korea"},
        "ms-SG": {"language": "Malay", "country": "Singapore"},
        "su-JB": {"language": "Sundanese", "country": "Indonesia"},
        "sv-SE": {"language": "Swedish", "country": "Sweden"},
        "ta-LK": {"language": "Tamil", "country": "Sri Lanka"},
        "ta-SG": {"language": "Tamil", "country": "Singapore"},
        "tl-PH": {"language": "Tagalog", "country": "Philippines"},
        "zh-CN": {"language": "Chinese", "country": "China"},
        "zh-SG": {"language": "Chinese", "country": "Singapore"},
        "zh-TW": {"language": "Chinese", "country": "Taiwan"},

        # “English parallel” locales
        "en-AS": {"language": "English", "country": "Assam"},
        "en-AZ": {"language": "English", "country": "Azerbaijan"},
        "en-BG": {"language": "English", "country": "Bulgaria"},
        "en-CN": {"language": "English", "country": "China"},
        "en-DZ": {"language": "English", "country": "Algeria"},
        "en-EC": {"language": "English", "country": "Ecuador"},
        "en-EG": {"language": "English", "country": "Egypt"},
        "en-ES": {"language": "English", "country": "Spain"},
        "en-ET": {"language": "English", "country": "Ethiopia"},
        "en-FR": {"language": "English", "country": "France"},
        "en-GR": {"language": "English", "country": "Greece"},
        "en-ID": {"language": "English", "country": "Indonesia"},
        "en-IE": {"language": "English", "country": "Ireland"},
        "en-IR": {"language": "English", "country": "Iran"},
        "en-JB": {"language": "English", "country": "Indonesia (West Java)"},
        "en-JP": {"language": "English", "country": "Japan"},
        "en-KP": {"language": "English", "country": "North Korea"},
        "en-KR": {"language": "English", "country": "South Korea"},
        "en-LK": {"language": "English", "country": "Sri Lanka"},
        "en-MA": {"language": "English", "country": "Morocco"},
        "en-MX": {"language": "English", "country": "Mexico"},
        "en-NG": {"language": "English", "country": "Nigeria"},
        "en-PH": {"language": "English", "country": "Philippines"},
        "en-PV": {"language": "English", "country": "Basque Country (Spain)"},
        "en-SA": {"language": "English", "country": "Saudi Arabia"},
        "en-SE": {"language": "English", "country": "Sweden"},
        "en-TW": {"language": "English", "country": "Taiwan"}
}




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



def split_id_locale(qid: str) -> Tuple[str, str, str]:
    """
    From an id like "ms-SG_001", return:
        locale = "ms-SG"
        lang = "ms"
        region = "SG"
    """
    locale = qid.split("_")[0]
    parts = locale.split("-")
    if len(parts) == 2:
        lang, region = parts
    else:
        # Fallback: just language, unknown region
        lang, region = parts[0], ""
    return locale, lang, region


def locale_info(id: str) -> Dict[str, str]:
    """
    Turn "ms-SG" into a small dict containing:
        {"locale": "ms-SG", "lang": "ms", "region": "SG",
         "lang_name": "Malay", "region_name": "Singapore"}
    """
    locale = id.split("_")[0]
    parts = locale.split("-")
    if len(parts) == 2:
        lang, region = parts
    else:
        lang, region = parts[0], ""
    return {
        "id":id,
        "locale": locale,
        "lang": lang,
        "region": region,
        "lang_name": language_and_country[locale]["language"], # LANG_NAME.get(region, lang),
        "region_name": language_and_country[locale]["country"], #REGION_NAME.get(region, region or "the relevant region"),
    }


# --------------------------- Prompt construction ---------------------------- #

def build_mcq_prompt(question: str,
                     option_a: str,
                     option_b: str,
                     option_c: str,
                     option_d: str,
                     locale_meta: Dict[str, str],
                     tokenizer, use_sys_prompt=False, 
                     enable_thinking=False) -> str:
    """
    Locale-aware MCQ prompt. We keep it as close as possible to
    your previous MCQ format, but with language/region hints.
    """
    ln = locale_meta["lang_name"]
    rn = locale_meta["region_name"]

    # return (
    #     f"You are answering a multiple-choice question for someone living in {rn}. "
    #     f"Respond strictly in {ln} and select exactly one option: A, B, C, or D.\n\n"
    #     f"Question: {question}\n"
    #     f"A. {option_a}\n"
    #     f"B. {option_b}\n"
    #     f"C. {option_c}\n"
    #     f"D. {option_d}\n"
    #     "Answer (A/B/C/D):"
    # )
    
    SYSTEM_PROMPT = (
        f"You are answering a multiple-choice question for someone living in {rn}. "
        f"Respond strictly in {ln} and select exactly one option: A, B, C, or D.\n\n"
        'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
    )

    # lang_promt = {
    # "es-ES": f"Selecciona exactamente una opción: A, B, C o D.\n\nPregunta: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nrespuesta:",
    # "ar-SA": f"اختر خيارًا واحدًا فقط: A أو B أو C أو D.\n\nالسؤال: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nالإجابة:",
    # "ja-JP": f"A、B、C、D の中から必ず 1 つ選択してください。\n\n質問: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n回答:",
    # "en-GB": f"Select exactly one option: A, B, C, or D.\n\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nanswer:",
    # "tl-PH": f"Pumili ng eksaktong isang opsyon: A, B, C, o D.\n\nTanong: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nsagot:",
    # "es-MX": f"Selecciona exactamente una opción: A, B, C o D.\n\nPregunta: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nrespuesta:",
    # "es-EC": f"Selecciona exactamente una opción: A, B, C o D.\n\nPregunta: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nrespuesta:",
    # "bg-BG": f"Избери точно една опция: A, B, C или D.\n\nВъпрос: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nотговор:",
    # "ko-KR": f"다음 중 정확히 하나의 선택지를 고르세요: A, B, C 또는 D.\n\n질문: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n답변:",
    # "id-ID": f"Pilih tepat satu opsi: A, B, C, atau D.\n\nPertanyaan: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\njawaban:",
    # "fa-IR": f"دقیقاً یک گزینه را انتخاب کنید: A، B، C یا D.\n\nسؤال: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nپاسخ:",
    # "ar-MA": f"اختر خيارًا واحدًا فقط: A أو B أو C أو D.\n\nالسؤال: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nالإجابة:",
    # "ga-IE": f"Roghnaigh rogha amháin go díreach: A, B, C nó D.\n\nCeist: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nfreagra:",
    # "fr-FR": f"Sélectionnez exactement une option : A, B, C ou D.\n\nQuestion : {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nréponse :",
    # "ar-EG": f"اختر خيارًا واحدًا فقط: A أو B أو C أو D.\n\nالسؤال: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nالإجابة:",
    # "ms-SG": f"Pilih tepat satu pilihan: A, B, C atau D.\n\nSoalan: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\njawapan:",
    # "eu-ES": f"Hautatu aukera bakarra zehazki: A, B, C edo D.\n\nGaldera: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nerantzuna:",
    # "ta-SG": f"A, B, C அல்லது D ஆகியவற்றில் ஒன்றை மட்டும் தேர்வு செய்யவும்.\n\nகேள்வி: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nபதில்:",
    # "en-AU": f"Select exactly one option: A, B, C, or D.\n\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nanswer:",
    # "zh-SG": f"请选择且仅选择一个选项：A、B、C 或 D。\n\n问题：{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n答案：",
    # "el-GR": f"Επιλέξτε ακριβώς μία επιλογή: A, B, C ή D.\n\nΕρώτηση: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nαπάντηση:",
    # "ta-LK": f"A, B, C அல்லது D ஆகியவற்றில் ஒன்றை மட்டும் தேர்வு செய்யவும்.\n\nகேள்வி: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nபதில்:",
    # "zh-CN": f"请选择且仅选择一个选项：A、B、C 或 D。\n\n问题：{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n答案："
    # }


    # user_text = lang_promt[locale_meta["locale"]] 

    user_text = (
        "Select exactly one option: A, B, C, or D.\n"
        f"Question: {question}\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"C. {option_c}\n"
        f"D. {option_d}\n"
        "answer:"
    )


    messages = []
    if use_sys_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    return prompt


def build_saq_prompt(question: str, locale_meta: Dict[str, str], tokenizer, use_sys_prompt=False, enable_thinking=False) -> str:
    """
    Locale-aware short-answer prompt.
    """
    ln = locale_meta["lang_name"]
    rn = locale_meta["region_name"]

    SYSTEM_PROMPT = (
        f"You are a knowledgeable local resident of {rn}. "
        f"Answer the following question in {ln} with a concise short answer "
        f"(one phrase or one short sentence) and do not provide any explanation.\n\n"
    )

    # lang_promt = {
    # "es-ES": f"Responde a la siguiente pregunta con una respuesta breve y concisa (una frase o una oración corta) y no proporciones ninguna explicación.\n\nPregunta: {question}\nRespuesta:",
    # "ar-SA": f"أجب عن السؤال التالي بإجابة قصيرة ومختصرة (عبارة واحدة أو جملة قصيرة) دون تقديم أي شرح.\n\nالسؤال: {question}\nالإجابة:",
    # "ja-JP": f"次の質問に対して、簡潔な短い回答（1つの語句または短い文）で答え、説明は一切行わないでください。\n\n質問: {question}\n回答:",
    # "en-GB": f"Answer the following question with a concise short answer (one phrase or one short sentence) and do not provide any explanation.\n\nQuestion: {question}\nAnswer:",
    # "tl-PH": f"Sagutin ang sumusunod na tanong gamit ang isang maikli at malinaw na sagot (isang parirala o isang maikling pangungusap) at huwag magbigay ng paliwanag.\n\nTanong: {question}\nSagot:",
    # "es-MX": f"Responde la siguiente pregunta con una respuesta breve y concisa (una frase o una oración corta) y no proporciones ninguna explicación.\n\nPregunta: {question}\nRespuesta:",
    # "es-EC": f"Responde la siguiente pregunta con una respuesta breve y concisa (una frase o una oración corta) y no proporciones ninguna explicación.\n\nPregunta: {question}\nRespuesta:",
    # "bg-BG": f"Отговори на следния въпрос с кратък и ясен отговор (една фраза или едно кратко изречение) и не давай никакво обяснение.\n\nВъпрос: {question}\nОтговор:",
    # "ko-KR": f"다음 질문에 대해 간결한 짧은 답변(한 구문 또는 한 문장)으로 답하고 설명은 제공하지 마세요.\n\n질문: {question}\n답변:",
    # "id-ID": f"Jawab pertanyaan berikut dengan jawaban singkat dan ringkas (satu frasa atau satu kalimat pendek) dan jangan berikan penjelasan apa pun.\n\nPertanyaan: {question}\nJawaban:",
    # "fa-IR": f"به پرسش زیر با یک پاسخ کوتاه و مختصر (یک عبارت یا یک جمله کوتاه) پاسخ دهید و هیچ توضیحی ارائه نکنید.\n\nسؤال: {question}\nپاسخ:",
    # "ar-MA": f"أجب عن السؤال التالي بإجابة قصيرة ومختصرة (عبارة واحدة أو جملة قصيرة) دون تقديم أي شرح.\n\nالسؤال: {question}\nالإجابة:",
    # "ga-IE": f"Freagair an cheist seo a leanas le freagra gairid agus beacht (frása amháin nó abairt ghearr) agus ná tabhair aon mhíniú.\n\nCeist: {question}\nFreagra:",
    # "fr-FR": f"Répondez à la question suivante par une réponse courte et concise (une phrase ou une courte phrase) et ne fournissez aucune explication.\n\nQuestion : {question}\nRéponse :",
    # "ar-EG": f"أجب عن السؤال التالي بإجابة قصيرة ومختصرة (عبارة واحدة أو جملة قصيرة) دون تقديم أي شرح.\n\nالسؤال: {question}\nالإجابة:",
    # "ms-SG": f"Jawab soalan berikut dengan jawapan ringkas dan padat (satu frasa atau satu ayat pendek) dan jangan berikan sebarang penjelasan.\n\nSoalan: {question}\nJawapan:",
    # "eu-ES": f"Erantzun hurrengo galdera erantzun labur eta zehatz batekin (esaldi bat edo esaldi labur bat) eta ez eman azalpenik.\n\nGaldera: {question}\nErantzuna:",
    # "ta-SG": f"பின்வரும் கேள்விக்கு சுருக்கமான குறுகிய பதிலுடன் (ஒரு சொற்றொடர் அல்லது ஒரு குறுகிய வாக்கியம்) பதிலளிக்கவும், எந்த விளக்கமும் வழங்க வேண்டாம்.\n\nகேள்வி: {question}\nபதில்:",
    # "en-AU": f"Answer the following question with a concise short answer (one phrase or one short sentence) and do not provide any explanation.\n\nQuestion: {question}\nAnswer:",
    # "zh-SG": f"请用简短的回答（一个短语或一句简短的话）回答以下问题，不要提供任何解释。\n\n问题：{question}\n答案：",
    # "el-GR": f"Απαντήστε στην ακόλουθη ερώτηση με μια σύντομη και περιεκτική απάντηση (μία φράση ή μία σύντομη πρόταση) και μην δώσετε καμία εξήγηση.\n\nΕρώτηση: {question}\nΑπάντηση:",
    # "ta-LK": f"பின்வரும் கேள்விக்கு சுருக்கமான குறுகிய பதிலுடன் (ஒரு சொற்றொடர் அல்லது ஒரு குறுகிய வாக்கியம்) பதிலளிக்கவும், எந்த விளக்கமும் வழங்க வேண்டாம்.\n\nகேள்வி: {question}\nபதில்:",
    # "zh-CN": f"请用简洁的回答（一个短语或一句简短的句子）回答以下问题，不要提供任何解释。\n\n问题：{question}\n答案："
    # }

    # user_text = lang_promt[locale_meta["locale"]]
    user_text = (
        f"Answer the following question with a concise short answer "
        f"(one phrase or one short sentence) and do not provide any explanation.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


    messages = []
    if use_sys_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Setting enable_thinking=False disables thinking mode
    )
    return prompt

    
# ------------------------------ Data loading -------------------------------- #

def load_track_saq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["locale"] = df["id"].apply(lambda x: split_id_locale(x)[0])
    df["lang"] = df["id"].apply(lambda x: split_id_locale(x)[1])
    df["region"] = df["id"].apply(lambda x: split_id_locale(x)[2])
    return df


def load_track_mcq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["locale"] = df["id"].apply(lambda x: split_id_locale(x)[0])
    df["lang"] = df["id"].apply(lambda x: split_id_locale(x)[1])
    df["region"] = df["id"].apply(lambda x: split_id_locale(x)[2])
    return df


def filter_locales(df: pd.DataFrame, lang_or_locale: str | None) -> List[str]:
    """
    If lang_or_locale is None: return all locales present in df.
    If it contains '-': treat as exact locale (e.g. 'en-GB').
    Else: treat as language code and return all locales for that language.
    """
    if lang_or_locale is None:
        return sorted(df["locale"].unique().tolist())

    if "-" in lang_or_locale:
        # Exact locale
        loc = lang_or_locale
        locales = sorted(df.loc[df["locale"] == loc, "locale"].unique().tolist())
    else:
        # Plain language: collect its locales
        lang = lang_or_locale
        locales = sorted(df.loc[df["lang"] == lang, "locale"].unique().tolist())
    return locales



def get_prompts(lang,
                tokenizer, 
                track: str, 
                split: str = "development", n: int = 200, 
                use_sys_prompt=False, 
                enable_thinking=False) -> List[Dict]:
    """Load BELEBELE dataset for reading comprehension from HuggingFace"""
    # Load data for the selected track
    if track == "mcq":
        df = load_track_mcq("data/eval_data/track_2_mcq_input.tsv")
        print(f"[data] MCQ loaded with {len(df)} rows.")
    else:
        df = load_track_saq("data/eval_data/track_1_saq_input.tsv")
        print(f"[data] SAQ loaded with {len(df)} rows.")
    # print(source_lang,target_lang, f"{lang_to_code(source_lang)}-{lang_to_code(target_lang)}")

    # locale2lang = get_flores_language_mapping()
    # lang2locale = {locale2lang[k]:k for k in locale2lang}

    

    df_locale = df[df["locale"] == lang].reset_index(drop=True)
    if df_locale.empty:
        print(f"[warn] No questions for locale {lang}. Skipping.")
        # print(df.head())


    prompts = []
    if track == "mcq":
        for i, row in df_locale.iterrows():
            meta = locale_info(row["id"])
            prompt = build_mcq_prompt(
                question=row["question"],
                option_a=row["option_A"],
                option_b=row["option_B"],
                option_c=row["option_C"],
                option_d=row["option_D"],
                locale_meta=meta,
                tokenizer=tokenizer,
                use_sys_prompt=use_sys_prompt,
                enable_thinking=enable_thinking
            )
            
            prompts.append((prompt,meta))
    else:
        for i, row in df_locale.iterrows():
            meta = locale_info(row["id"])
            prompt = build_saq_prompt(row["question"], meta,
                tokenizer=tokenizer,
                use_sys_prompt=use_sys_prompt,
                enable_thinking=enable_thinking
            )
            prompts.append((prompt,meta))


    return prompts 





# def clean_results(text, track):
#     if track == "mcq":
#         choices = ["A", "B", "C", "D"]
#         text = text.replace("answer","")
#         out = {}
#         for c in choices:
#             if c in text:
#                 out[c] = 1
#             else:
#                 out[c] = 0
#     else:
#         out = text
#     return out


import re

def clean_results(text: str, track: str):
    if track != "mcq":
        return text

    choices = ["A", "B", "C", "D"]
    text_norm = text.strip().upper()

    # 1) Strong patterns (highest confidence)
    strong_patterns = [
        r"\bANSWER\s*[:\-]?\s*([ABCD])\b",
        r"\bCORRECT\s+ANSWER\s*[:\-]?\s*([ABCD])\b",
        r"\bTHE\s+ANSWER\s+IS\s+([ABCD])\b",
        r"\bOPTION\s+([ABCD])\b",
        r"\(([ABCD])\)",
    ]

    for pattern in strong_patterns:
        match = re.search(pattern, text_norm)
        if match:
            return {c: int(c == match.group(1)) for c in choices}

    # 2) Standalone single-letter answer (medium confidence)
    standalone = re.findall(r"\b([ABCD])\b", text_norm)
    if len(standalone) == 1:
        return {c: int(c == standalone[0]) for c in choices}

    # 3) Fallback: first occurrence in text order (low confidence)
    for c in choices:
        if re.search(rf"\b{c}\b", text_norm):
            return {x: int(x == c) for x in choices}

    # 4) Explicit failure case
    return {c: 0 for c in choices}



def post_process_text(text):

    text = text.split("</think>\n\n")[1] if len(text.split("</think>\n\n")) > 0 else text
    
    text = text.split("<|end_of_text|>")[0] if len(text.split("<|end_of_text|>")) > 0 else text
    #text = text.split("?")[0] if len(text.split("?")) > 0 else text
    text = text.split("\n")[0] if len(text.split("\n")) > 0 else text
    #text = text.split(".")[0] if len(text.split(".")) > 0 else text
    # Strip excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #text = re.sub(r'[?]+', '?', text).strip()
    text = re.sub(r'(\?){2,}', '?', text).strip()

    return text

