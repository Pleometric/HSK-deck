#!/usr/bin/env python3
"""
Generate Anki deck from cleaned vocabulary with constrained sentence generation.
Supports parallelization for fast generation.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import genanki
import random

try:
    import edge_tts
except ImportError:
    edge_tts = None

# Load environment variables
load_dotenv()

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
CLEANED_DIR = SCRIPT_DIR / "data/cleaned"
DATA_DIR = SCRIPT_DIR / "data"
AUDIO_ROOT_DIR = SCRIPT_DIR / "audio"
WORD_AUDIO_DIR = AUDIO_ROOT_DIR / "words"
SENTENCE_AUDIO_DIR = AUDIO_ROOT_DIR / "sentences"
DECK_DIR = SCRIPT_DIR / "decks"

WORD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SENTENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
DECK_DIR.mkdir(parents=True, exist_ok=True)

# Audio configuration
MICROSOFT_TTS_VOICE = os.getenv('MICROSOFT_TTS_VOICE', 'zh-CN-YunjianNeural')
MICROSOFT_TTS_RATE = os.getenv('MICROSOFT_TTS_RATE', '+0%')
MICROSOFT_TTS_VOLUME = os.getenv('MICROSOFT_TTS_VOLUME', '+0%')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', 'pTOe8BQRdydOEIgv0wFL')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')

# API Clients
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '').strip()
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Seed vocabulary (always allowed in sentences)
SEED_VOCABULARY = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "谁",
    "是", "有", "做", "去", "来", "看", "说", "听", "吃", "喝", "买", "要", "想", "能", "会",
    "的", "了", "着", "过", "得", "地", "在", "和", "跟", "也", "都", "很", "不", "没", "吗", "呢", "吧", "就", "还", "又",
    "什么", "哪", "怎么", "为什么", "多少", "几", "哪里", "怎样",
    "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万",
    "大", "小", "好", "多", "少", "今天", "昨天", "明天", "现在", "这里"
}

# Sentence generation prompt (full original + pinyin specification)
SENTENCE_SYSTEM_PROMPT = """任务：为目标词生成包含该词的中文例句，并提供拼音和英文翻译。

=== CRITICAL约束 ===
目标词必须完整出现在句子中。
- 不可用同义词或近义词替换
- 如果目标词是单字，必须作为独立的字使用，不能仅作为复合词的一部分
- 每次生成必须不同，严禁重复相同的句子
- **必须使用指定的拼音读音和含义**（例如：如果指定"了"读作"le"，必须使用"le"的用法，不能使用"liǎo"的含义）

单字示例：
✓ 目标词"无" → "这个问题无人能回答" (无作为独立字)
✗ 目标词"无" → "我很无聊" (无只在复合词"无聊"中)

词汇选择规则：
- 优先使用词汇表中的词（词汇表按学习时间排序，**末尾是最近学的词，优先选择**）
- 可使用常见虚词和量词（的、了、着、过、在、是、有、被、把、得、地等）
- 避免使用词汇表外的实词

语域匹配：
- 根据目标词的语言特征选择适当的语域
- 书面语/正式词汇 → 正式语境（新闻、公告、学术）
- 口语/日常词汇 → 自然、口语化语境
- **必须使用现代汉语语法，不使用文言文句式**

成对语法结构（必须完整）：
如目标词是以下结构之一，必须包含配对部分：
与其...不如... / 不但...而且... / 虽然...但是... / 因为...所以... / 既...又... / 一边...一边... / 不仅...还... / 要么...要么... / 既然...就... / 只有...才... / 只要...就... / 无论...都... / 越...越... / 宁可...也不... / 如果...就... / 尽管...还是... / 不是...而是... / 先...再... / 一...就... / 又...又...

句子要求：
- 8-20个汉字
- 语法完整、标点正确、现代汉语
- 中级语法结构
- 展示目标词在指定读音下的最典型用法

输出格式（JSON）：
返回包含以下字段的JSON对象：
{
  "sentence": "完整的中文例句",
  "sentence_pinyin": "完整句子的拼音（带声调标记）",
  "sentence_translation": "英文翻译"
}

示例：
输入: {"target": "了", "pinyin": "le", "meanings": ["aspect particle"], "vocabulary": ["我", "吃", "饭"]}
输出: {"sentence": "我吃了饭。", "sentence_pinyin": "Wǒ chī le fàn.", "sentence_translation": "I ate."}

输入: {"target": "了", "pinyin": "liǎo", "meanings": ["to finish"], "vocabulary": ["这", "事", "我"]}
输出: {"sentence": "这件事我了不了。", "sentence_pinyin": "Zhè jiàn shì wǒ liǎo bù liǎo.", "sentence_translation": "I can't finish this matter."}

输入: {"target": "苹果", "pinyin": "píngguǒ", "meanings": ["apple"], "vocabulary": ["我", "买", "超市", "新鲜"]}
输出: {"sentence": "超市里的苹果很新鲜。", "sentence_pinyin": "Chāoshì lǐ de píngguǒ hěn xīnxiān.", "sentence_translation": "The apples in the supermarket are very fresh."}"""


def extract_json_from_response(text: str) -> Dict:
    """Extract JSON from DeepSeek response."""
    # Try 1: Direct parsing
    try:
        return json.loads(text)
    except:
        pass

    # Try 2: Extract from markdown
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Try 3: Find JSON object
    match = re.search(r'\{[\s\S]*?"sentence"[\s\S]*?\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {}


def build_known_vocabulary(all_cards: List[Dict], up_to_index: int) -> Set[str]:
    """Build known vocabulary up to (not including) the given index."""
    known = set(SEED_VOCABULARY)
    for i in range(up_to_index):
        known.add(all_cards[i]['word'])
    return known


def generate_sentence_for_card(card: Dict, known_vocab: Set[str], card_idx: int) -> Dict:
    """Generate sentence for a single card. Returns card with sentence data added."""

    vocab_list = sorted(list(known_vocab))

    user_message = json.dumps({
        "target": card['word'],
        "pinyin": card['pinyin'],
        "meanings": card['meanings'],
        "vocabulary": vocab_list
    }, ensure_ascii=False)

    for attempt in range(3):
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SENTENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )

            result = response.choices[0].message.content.strip()
            parsed = extract_json_from_response(result)

            if parsed and 'sentence' in parsed:
                card['sentence'] = parsed.get('sentence', '')
                card['sentence_pinyin'] = parsed.get('sentence_pinyin', '')
                card['sentence_translation'] = parsed.get('sentence_translation', '')
                return card

            if attempt < 2:
                time.sleep(1)

        except Exception as e:
            if attempt < 2:
                time.sleep(2)

    # Fallback
    card['sentence'] = card['word']
    card['sentence_pinyin'] = card['pinyin']
    card['sentence_translation'] = '; '.join(card['meanings'])
    return card


def microsoft_tts_available() -> bool:
    """Return True when the Microsoft Edge TTS dependency is installed."""
    return edge_tts is not None


def audio_setup_is_valid() -> bool:
    """Return True when at least one sentence provider and one word provider are available."""
    word_provider_available = microsoft_tts_available() or elevenlabs_client is not None
    sentence_provider_available = elevenlabs_client is not None or microsoft_tts_available()
    return word_provider_available and sentence_provider_available


def word_audio_provider_label() -> str:
    """Describe the current word audio provider."""
    if microsoft_tts_available():
        return f"Microsoft Edge Neural ({MICROSOFT_TTS_VOICE})"
    if elevenlabs_client is not None:
        return "ElevenLabs fallback"
    return "Unavailable"


def sentence_audio_provider_label() -> str:
    """Describe the current sentence audio provider."""
    if elevenlabs_client is not None:
        return "ElevenLabs"
    if microsoft_tts_available():
        return f"Microsoft Edge Neural fallback ({MICROSOFT_TTS_VOICE})"
    return "Unavailable"


async def synthesize_microsoft_audio(text: str, output_path: Path) -> None:
    """Generate a single MP3 via Microsoft Edge neural voices."""
    tts = edge_tts.Communicate(
        text=text,
        voice=MICROSOFT_TTS_VOICE,
        rate=MICROSOFT_TTS_RATE,
        volume=MICROSOFT_TTS_VOLUME,
    )
    await tts.save(str(output_path))


def generate_microsoft_audio(text: str, output_path: Path, retry_count: int = 3) -> bool:
    """Generate audio using Microsoft Edge neural TTS with retries."""
    if not microsoft_tts_available():
        return False

    for attempt in range(retry_count):
        try:
            asyncio.run(synthesize_microsoft_audio(text, output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            if output_path.exists():
                output_path.unlink()
        except Exception:
            if output_path.exists():
                output_path.unlink()
            if attempt < retry_count - 1:
                time.sleep(1 + attempt)

    return False


def generate_elevenlabs_audio(text: str, output_path: Path, retry_count: int = 3) -> bool:
    """Generate audio using ElevenLabs with retries."""
    if elevenlabs_client is None:
        return False

    for attempt in range(retry_count):
        try:
            audio_generator = elevenlabs_client.text_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                text=text,
                model_id=ELEVENLABS_MODEL_ID
            )

            with open(output_path, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)

            if output_path.exists() and output_path.stat().st_size > 0:
                return True

            if output_path.exists():
                output_path.unlink()

        except Exception as e:
            if output_path.exists():
                output_path.unlink()

            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                if attempt < retry_count - 1:
                    time.sleep((2 ** attempt) * 0.5)
                    continue

            if attempt < retry_count - 1:
                time.sleep(1)

            break

    return False


def generate_audio_filename(card: Dict, card_idx: int, suffix: str) -> str:
    """Build the stable audio filename for a card."""
    return f"hsk{card['hsk_level']}_{card_idx:05d}_{suffix}.mp3"


def generate_sentences_batch(all_cards: List[Dict], batch_start: int, batch_size: int, max_workers: int = 10) -> None:
    """Generate sentences for a batch of cards in parallel."""

    batch_end = min(batch_start + batch_size, len(all_cards))
    batch = all_cards[batch_start:batch_end]

    # All cards in this batch use the same known vocabulary (up to batch_start)
    known_vocab = build_known_vocabulary(all_cards, batch_start)

    print(f"\nBatch {batch_start}-{batch_end-1} ({len(batch)} cards, {len(known_vocab)} known words)")

    # Generate sentences in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for local_idx, card in enumerate(batch):
            card_idx = batch_start + local_idx
            future = executor.submit(generate_sentence_for_card, card, known_vocab, card_idx)
            futures[future] = card_idx

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == len(batch):
                print(f"  Progress: {completed}/{len(batch)}")


def generate_audio_for_card(card: Dict, card_idx: int) -> Dict:
    """Generate word and sentence audio for a single card."""

    word_audio_filename = generate_audio_filename(card, card_idx, "word")
    word_audio_path = WORD_AUDIO_DIR / word_audio_filename

    if word_audio_path.exists() and word_audio_path.stat().st_size > 0:
        card['word_audio_file'] = word_audio_filename
    else:
        word_generated = False
        if microsoft_tts_available():
            word_generated = generate_microsoft_audio(card['word'], word_audio_path)
        if not word_generated and elevenlabs_client is not None:
            word_generated = generate_elevenlabs_audio(card['word'], word_audio_path)

        card['word_audio_file'] = word_audio_filename if word_generated else ""

    sentence_audio_filename = generate_audio_filename(card, card_idx, "sentence")
    sentence_audio_path = SENTENCE_AUDIO_DIR / sentence_audio_filename

    if sentence_audio_path.exists() and sentence_audio_path.stat().st_size > 0:
        card['sentence_audio_file'] = sentence_audio_filename
        return card

    sentence_generated = False
    if elevenlabs_client is not None:
        sentence_generated = generate_elevenlabs_audio(card['sentence'], sentence_audio_path)
    if not sentence_generated and microsoft_tts_available():
        sentence_generated = generate_microsoft_audio(card['sentence'], sentence_audio_path)

    card['sentence_audio_file'] = sentence_audio_filename if sentence_generated else ""
    return card


def generate_audio_parallel(all_cards: List[Dict], max_workers: int = 5) -> None:
    """Generate word and sentence audio files in parallel."""

    print(f"\nGenerating audio with {max_workers} parallel workers...")
    print(f"  Word audio provider: {word_audio_provider_label()}")
    print(f"  Sentence audio provider: {sentence_audio_provider_label()}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, card in enumerate(all_cards):
            future = executor.submit(generate_audio_for_card, card, idx)
            futures[future] = idx

            # Rate limit: small delay between submissions
            if (idx + 1) % max_workers == 0:
                time.sleep(0.1)

        completed = 0
        word_successful = 0
        sentence_successful = 0
        for future in as_completed(futures):
            idx = futures[future]
            updated_card = future.result()
            all_cards[idx] = updated_card  # Update the card in the list

            if updated_card.get('word_audio_file'):
                word_successful += 1
            if updated_card.get('sentence_audio_file'):
                sentence_successful += 1

            completed += 1
            if completed % 50 == 0 or completed == len(all_cards):
                print(
                    f"  Progress: {completed}/{len(all_cards)} ({completed/len(all_cards)*100:.1f}%)"
                    f" - {word_successful} words, {sentence_successful} sentences"
                )




def create_anki_model():
    """Create Anki note model for cards."""
    model_id = random.randrange(1 << 30, 1 << 31)

    return genanki.Model(
        model_id,
        'HSK Chinese Card',
        fields=[
            {'name': 'Word'},
            {'name': 'Pinyin'},
            {'name': 'Meaning'},
            {'name': 'Sentence'},
            {'name': 'SentencePinyin'},
            {'name': 'SentenceTranslation'},
            {'name': 'Classifier'},
            {'name': 'HSKLevel'},
            {'name': 'WordAudio'},
            {'name': 'SentenceAudio'},
        ],
        templates=[
            {
                'name': 'Card',
                'qfmt': '''
<div class="chinese-card">
  <div class="word">{{Word}}</div>
  <div class="sentence">{{Sentence}}</div>
</div>

<script>
(function() {
  const word = document.querySelector('.word').textContent.trim();
  const sentenceDiv = document.querySelector('.sentence');
  const sentenceRaw = sentenceDiv.innerHTML.trim();

  if (sentenceRaw.includes('{') && sentenceRaw.includes('}')) {
    sentenceDiv.innerHTML = sentenceRaw.replace(
      /\\{([^}]+)\\}/g,
      '<span class="target">$1</span>'
    );
  } else if (word && sentenceRaw.includes(word)) {
    sentenceDiv.innerHTML = sentenceRaw.replace(
      new RegExp(word, 'g'),
      '<span class="target">' + word + '</span>'
    );
  }
})();
</script>

<style>
.card {
  background: #1e1e1e;
  color: #e0e0e0;
  font-family: "Noto Sans SC", sans-serif;
}

.chinese-card {
  background: #2a2a2a;
  padding: 20px;
  max-width: 800px;
  margin: 20px auto;
  border-radius: 8px;
}

.word { font-size: 72px; margin: 20px 0; text-align: center; }
.sentence { font-size: 26px; margin: 30px 0; line-height: 1.8; text-align: center; }
.sentence .target { color: #4ade80; font-weight: bold; }

.back { margin-top: 20px; }
.pinyin { font-size: 24px; color: #888; margin: 10px 0; text-align: center; }
.sentence-pinyin { font-size: 18px; color: #666; margin: 10px 0; font-style: italic; text-align: center; }
.classifier { font-size: 16px; color: #666; margin: 15px 0; text-align: center; }

.button-container { text-align: center; margin: 15px 0; }
.reveal-btn {
  padding: 10px 20px;
  background: #4a4a4a;
  color: #e0e0e0;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
  transition: background 0.2s;
}
.reveal-btn:hover { background: #5a5a5a; }

.hidden-content {
  display: none;
  margin: 15px 0;
  font-size: 18px;
  color: #b0b0b0;
  text-align: center;
}

.audio-hidden { display: none; }
hr { border: none; border-top: 1px solid #444; margin: 20px 0; }

.tone1 { color: #ff4d4f !important; }
.tone2 { color: #ff9800 !important; }
.tone3 { color: #4caf50 !important; }
.tone4 { color: #42a5f5 !important; }
.tone5 { color: #9e9e9e !important; }
</style>
<div class="audio-hidden">{{WordAudio}}</div>
''',
                'afmt': '''
{{FrontSide}}

<hr>

<div class="back">
  <div class="pinyin">{{Pinyin}}</div>
  <div class="sentence-pinyin">{{SentencePinyin}}</div>

  <div class="button-container">
    <button onclick="document.getElementById('sent-trans').style.display='block'; this.style.display='none';" class="reveal-btn">
      Show Sentence Translation
    </button>
  </div>
  <div id="sent-trans" class="hidden-content">
    {{SentenceTranslation}}
  </div>

  <div class="button-container">
    <button onclick="document.getElementById('meaning').style.display='block'; this.style.display='none';" class="reveal-btn">
      Show Word Meaning
    </button>
  </div>
  <div id="meaning" class="hidden-content">
    {{Meaning}}
  </div>

  {{#Classifier}}<div class="classifier">{{Classifier}}</div>{{/Classifier}}
  <div class="audio-hidden">{{SentenceAudio}}</div>
</div>

<script>
(function() {
    var TONE_MARK_TO_TONE = {
        "ā": 1, "á": 2, "ǎ": 3, "à": 4,
        "ē": 1, "é": 2, "ě": 3, "è": 4,
        "ī": 1, "í": 2, "ǐ": 3, "ì": 4,
        "ō": 1, "ó": 2, "ǒ": 3, "ò": 4,
        "ū": 1, "ú": 2, "ǔ": 3, "ù": 4,
        "ǖ": 1, "ǘ": 2, "ǚ": 3, "ǜ": 4
    };
    var COMBINING_MARK_TO_TONE = {
        "\\u0304": 1, "\\u0301": 2, "\\u030c": 3, "\\u0300": 4
    };
    var TONE_COLORS = {
        1: "#ff4d4f", 2: "#ff9800", 3: "#4caf50", 4: "#42a5f5", 5: "#9e9e9e"
    };
    var BASE_VOWEL_TO_MARKED = {
        "a": ["ā", "á", "ǎ", "à"],
        "e": ["ē", "é", "ě", "è"],
        "i": ["ī", "í", "ǐ", "ì"],
        "o": ["ō", "ó", "ǒ", "ò"],
        "u": ["ū", "ú", "ǔ", "ù"],
        "ü": ["ǖ", "ǘ", "ǚ", "ǜ"],
        "A": ["Ā", "Á", "Ǎ", "À"],
        "E": ["Ē", "É", "Ě", "È"],
        "I": ["Ī", "Í", "Ǐ", "Ì"],
        "O": ["Ō", "Ó", "Ǒ", "Ò"],
        "U": ["Ū", "Ú", "Ǔ", "Ù"],
        "Ü": ["Ǖ", "Ǘ", "Ǚ", "Ǜ"]
    };
    var TONE_MARK_REGEX = /[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ]/;
    var PINYIN_CHARS = "A-Za-züÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ";
    var PINYIN_RUN_REGEX = new RegExp("^[" + PINYIN_CHARS + "]+$");
    var PINYIN_TOKEN_REGEX = new RegExp("[" + PINYIN_CHARS + "]+|\\\\s+|[^" + PINYIN_CHARS + "\\\\s]+", "g");
    var POST_TONE_SEGMENT_REGEX = /^(?:[aeiouü]{0,2}(?:n|ng|r)?)$/;
    var PRE_TONE_SEGMENT_REGEX = /^(?:(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?[aeiouü]{0,2})$/;
    var PINYIN_INITIAL_FINAL_REGEX = /^(?:(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsl])(?:a|o|e|i|u|ai|ei|ao|ou|an|en|ang|eng|ong|ia|ie|iao|iu|ian|in|iang|ing|iong|ua|uo|uai|ui|uan|un|uang|ueng|ü|üe|üan|ün))$/;
    var PINYIN_SPECIAL_REGEX = /^(?:zhi|chi|shi|ri|zi|ci|si|er|yi|ya|yo|ye|yao|you|yan|yin|yang|ying|yong|yu|yue|yuan|yun|wu|wa|wo|wai|wei|wan|wen|wang|weng|a|o|e|ai|ei|ao|ou|an|en|ang|eng|ong|m|n|ng)$/;

    function toBasePinyin(text) {
        return Array.from(text).map(function(char) {
            var lower = char.toLowerCase();
            if (TONE_MARK_TO_TONE.hasOwnProperty(lower)) {
                if ("āáǎà".includes(lower)) return "a";
                if ("ēéěè".includes(lower)) return "e";
                if ("īíǐì".includes(lower)) return "i";
                if ("ōóǒò".includes(lower)) return "o";
                if ("ūúǔù".includes(lower)) return "u";
                if ("ǖǘǚǜ".includes(lower)) return "ü";
            }
            if (lower === "ü") return "ü";
            return lower;
        }).join("");
    }

    function normalizePinyinText(text) {
        var normalized = (text || "")
            .replace(/u:/g, "ü")
            .replace(/U:/g, "Ü")
            .replace(/v/g, "ü")
            .replace(/V/g, "Ü")
            .replace(/[·•・]/g, " ")
            .replace(/\\s+/g, " ")
            .trim();
        normalized = normalized.replace(/[A-Za-züÜ]+[1-5]/g, function(match) {
            return convertNumberedSyllable(match);
        });
        return normalized.normalize ? normalized.normalize("NFC") : normalized;
    }

    function addToneMark(syllable, tone) {
        if (tone === 5) return syllable;

        var lower = syllable.toLowerCase();
        var markIndex = -1;

        if (lower.indexOf("a") !== -1) {
            markIndex = lower.indexOf("a");
        } else if (lower.indexOf("e") !== -1) {
            markIndex = lower.indexOf("e");
        } else if (lower.indexOf("ou") !== -1) {
            markIndex = lower.indexOf("o");
        } else {
            for (var i = syllable.length - 1; i >= 0; i--) {
                if ("aeiouüAEIOUÜ".indexOf(syllable[i]) !== -1) {
                    markIndex = i;
                    break;
                }
            }
        }

        if (markIndex === -1) return syllable;

        var marked = BASE_VOWEL_TO_MARKED[syllable[markIndex]];
        if (!marked) return syllable;

        return syllable.slice(0, markIndex) + marked[tone - 1] + syllable.slice(markIndex + 1);
    }

    function convertNumberedSyllable(syllable) {
        var match = syllable.match(/^([A-Za-züÜ]+)([1-5])$/);
        if (!match) return syllable;

        var base = match[1];
        var tone = Number(match[2]);
        return addToneMark(base, tone);
    }

    function detectTone(syllable) {
        for (var i = 0; i < syllable.length; i++) {
            var char = syllable[i];
            var tone = TONE_MARK_TO_TONE[char.toLowerCase()];
            if (tone) return tone;
            if (COMBINING_MARK_TO_TONE.hasOwnProperty(char)) return COMBINING_MARK_TO_TONE[char];
        }
        return 5;
    }

    function isLikelyPinyinSyllable(syllable) {
        var base = toBasePinyin(syllable);
        return PINYIN_SPECIAL_REGEX.test(base) || PINYIN_INITIAL_FINAL_REGEX.test(base);
    }

    function chooseSplitBetweenTones(run, start, toneIndex, nextToneIndex) {
        for (var split = nextToneIndex; split >= toneIndex + 1; split--) {
            var leftTail = toBasePinyin(run.slice(toneIndex + 1, split));
            var rightHead = toBasePinyin(run.slice(split, nextToneIndex));
            var leftSyllable = run.slice(start, split);
            if (POST_TONE_SEGMENT_REGEX.test(leftTail) && PRE_TONE_SEGMENT_REGEX.test(rightHead) && isLikelyPinyinSyllable(leftSyllable)) return split;
        }
        return toneIndex + 1;
    }

    function chooseTailSplit(run, start, toneIndex) {
        var trailing = run.slice(toneIndex + 1);
        for (var len = trailing.length; len >= 0; len--) {
            var tail = toBasePinyin(trailing.slice(0, len));
            var end = toneIndex + 1 + len;
            var syllable = run.slice(start, end);
            if (POST_TONE_SEGMENT_REGEX.test(tail) && isLikelyPinyinSyllable(syllable)) return end;
        }
        return run.length;
    }

    function alignSyllableStart(run, start, toneIndex) {
        for (var candidate = start; candidate <= toneIndex; candidate++) {
            var head = toBasePinyin(run.slice(candidate, toneIndex));
            if (PRE_TONE_SEGMENT_REGEX.test(head)) return candidate;
        }
        return start;
    }

    function splitPinyinRun(run) {
        var toneIndexes = [];
        for (var i = 0; i < run.length; i++) {
            if (TONE_MARK_REGEX.test(run[i])) toneIndexes.push(i);
        }
        if (toneIndexes.length === 0) return [run];

        var syllables = [];
        var start = 0;
        for (var i = 0; i < toneIndexes.length; i++) {
            var toneIndex = toneIndexes[i];
            var alignedStart = alignSyllableStart(run, start, toneIndex);
            if (alignedStart > start) {
                syllables.push(run.slice(start, alignedStart));
                start = alignedStart;
            }
            if (i < toneIndexes.length - 1) {
                var end = chooseSplitBetweenTones(run, start, toneIndex, toneIndexes[i + 1]);
                syllables.push(run.slice(start, end));
                start = end;
            } else {
                var end = chooseTailSplit(run, start, toneIndex);
                syllables.push(run.slice(start, end));
                if (end < run.length) syllables.push(run.slice(end));
            }
        }
        return syllables.filter(Boolean);
    }

    function colorizePinyin(element) {
        if (!element) return;
        var reading = normalizePinyinText(element.textContent);
        if (!reading) return;

        var tokens = reading.match(PINYIN_TOKEN_REGEX) || [];
        var fragment = document.createDocumentFragment();
        for (var t = 0; t < tokens.length; t++) {
            var token = tokens[t];
            if (!PINYIN_RUN_REGEX.test(token)) {
                fragment.appendChild(document.createTextNode(token));
                continue;
            }
            var syllables = splitPinyinRun(token);
            for (var s = 0; s < syllables.length; s++) {
                var syllable = syllables[s];
                var tone = detectTone(syllable);
                var span = document.createElement("span");
                span.className = "tone-syllable tone" + tone;
                span.textContent = syllable;
                span.style.color = TONE_COLORS[tone] || TONE_COLORS[5];
                fragment.appendChild(span);
            }
        }
        element.textContent = "";
        element.appendChild(fragment);
    }

    colorizePinyin(document.querySelector('.back .pinyin'));
})();
</script>
'''
            }
        ]
    )


def generate_deck(levels: List[int], batch_size: int = 50, sentence_workers: int = 10, audio_workers: int = 5):
    """Generate complete Anki deck from cleaned vocabulary."""

    print(f"\n{'='*80}")
    print(f"GENERATING DECK FOR HSK LEVELS: {levels}")
    print(f"{'='*80}")
    print(f"Parallelization: {sentence_workers} sentence workers, {audio_workers} audio workers")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")

    if not audio_setup_is_valid():
        print("❌ No usable audio setup found.")
        print("   Install Microsoft TTS support with: uv pip install edge-tts")
        print("   Or configure ELEVENLABS_API_KEY for ElevenLabs sentence audio.")
        return

    # Load all cleaned vocabulary
    all_cards = []
    for level in levels:
        cleaned_file = CLEANED_DIR / f"{level}_cleaned.json"
        if not cleaned_file.exists():
            print(f"❌ Missing cleaned file: {cleaned_file}")
            print(f"   Run: uv run clean_vocabulary.py")
            return

        with open(cleaned_file, 'r', encoding='utf-8') as f:
            cards = json.load(f)
            all_cards.extend(cards)

    # Sort by frequency across all levels
    all_cards.sort(key=lambda x: x.get('frequency', 999999))
    print(f"Total cards to process: {len(all_cards)}\n")

    # Stage 1: Generate sentences
    print(f"{'='*80}")
    print("STAGE 1: GENERATING SENTENCES (PARALLEL)")
    print(f"{'='*80}")

    # Process in batches
    for batch_start in range(0, len(all_cards), batch_size):
        generate_sentences_batch(all_cards, batch_start, batch_size, sentence_workers)
        # Rate limiting between batches
        time.sleep(2)

    print(f"\n✓ Completed sentence generation\n")

    # Stage 2: Generate audio
    print(f"{'='*80}")
    print("STAGE 2: GENERATING AUDIO (PARALLEL)")
    print(f"{'='*80}")

    generate_audio_parallel(all_cards, audio_workers)

    print(f"\n✓ Completed audio generation\n")

    # Stage 3: Build Anki deck
    print(f"{'='*80}")
    print("STAGE 3: BUILDING ANKI DECK")
    print(f"{'='*80}\n")

    # Create deck and subdecks
    deck_name = "HSK 3.0 Mandarin (Cleaned)"
    parent_deck_id = random.randrange(1 << 30, 1 << 31)

    model = create_anki_model()
    subdecks = {}

    for level in set(card['hsk_level'] for card in all_cards):
        subdeck_name = f"{deck_name}::HSK {level}"
        subdeck_id = parent_deck_id + level
        subdecks[level] = genanki.Deck(subdeck_id, subdeck_name)
        print(f"  Created subdeck: {subdeck_name}")

    # Add notes to subdecks
    media_files = []

    for card in all_cards:
        note = genanki.Note(
            model=model,
            fields=[
                card['word'],
                card['pinyin'],
                '; '.join(card['meanings']),
                card.get('sentence', card['word']),
                card.get('sentence_pinyin', card['pinyin']),
                card.get('sentence_translation', ''),
                card.get('classifier', ''),
                str(card['hsk_level']),
                f"[sound:{card['word_audio_file']}]" if card.get('word_audio_file') else "",
                f"[sound:{card['sentence_audio_file']}]" if card.get('sentence_audio_file') else "",
            ]
        )

        subdecks[card['hsk_level']].add_note(note)

        if card.get('word_audio_file'):
            word_audio_path = WORD_AUDIO_DIR / card['word_audio_file']
            if word_audio_path.exists():
                media_files.append(str(word_audio_path))

        if card.get('sentence_audio_file'):
            sentence_audio_path = SENTENCE_AUDIO_DIR / card['sentence_audio_file']
            if sentence_audio_path.exists():
                media_files.append(str(sentence_audio_path))

    print(f"\n✓ Added {len(all_cards)} notes to deck")
    print(f"✓ Collected {len(media_files)} audio files\n")

    # Package deck
    level_range = f"{min(levels)}-{max(levels)}" if len(levels) > 1 else str(levels[0])
    output_file = DECK_DIR / f"HSK_{level_range}_cleaned.apkg"

    package = genanki.Package(list(subdecks.values()))
    package.media_files = media_files
    package.write_to_file(output_file)

    print(f"{'='*80}")
    print(f"✅ DECK GENERATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nOutput: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Cards: {len(all_cards)}")
    print(f"Audio files: {len(media_files)}")


def main():
    print("="*80)
    print("HSK DECK GENERATOR")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Parallel sentence generation (batch processing)")
    print("  ✓ Microsoft neural word audio")
    print("  ✓ ElevenLabs sentence audio when configured")
    print("  ✓ Microsoft neural fallback for sentence audio")
    print("="*80)

    level_input = input("\nWhich HSK level(s)? [1-7 or '1,2,3']: ").strip()

    try:
        levels = [int(x.strip()) for x in level_input.split(',')]
        if not all(1 <= level <= 7 for level in levels):
            print("❌ Invalid levels")
            return
    except ValueError:
        print("❌ Invalid input")
        return

    # Check cleaned files exist
    missing = []
    for level in levels:
        if not (CLEANED_DIR / f"{level}_cleaned.json").exists():
            missing.append(level)

    if missing:
        print(f"\n⚠ Missing cleaned vocabulary for levels: {missing}")
        print(f"\nThe vocabulary cleaner uses DeepSeek to intelligently:")
        print(f"  - Select correct pronunciations (le vs liǎo)")
        print(f"  - Create multiple cards for multi-pronunciation words")
        print(f"  - Skip rare/surname/literary forms")
        print(f"\nEstimated time: ~5-10 minutes")

        auto_clean = input(f"\nAuto-run cleaner for levels {missing}? (yes/no) [yes]: ").strip().lower()

        if auto_clean in ['', 'y', 'yes']:
            import subprocess
            print(f"\n{'='*80}")
            print(f"RUNNING VOCABULARY CLEANER")
            print(f"{'='*80}\n")

            # Run clean_vocabulary.py in non-interactive mode
            missing_str = ','.join(map(str, missing))
            try:
                result = subprocess.run(
                    ['uv', 'run', 'clean_vocabulary.py', missing_str],
                    check=True,
                    cwd=SCRIPT_DIR
                )

                if result.returncode == 0:
                    print(f"\n✓ Vocabulary cleaning complete!")
                else:
                    print(f"\n❌ Cleaner failed with exit code {result.returncode}")
                    return

            except Exception as e:
                print(f"\n❌ Error running cleaner: {e}")
                return

        else:
            print(f"\n❌ Cannot proceed without cleaned vocabulary")
            print(f"   Run: uv run clean_vocabulary.py")
            return

    print(f"\nGenerating deck for levels: {levels}")
    input("\nPress Enter to start...")

    generate_deck(levels, batch_size=50, sentence_workers=10, audio_workers=5)


if __name__ == "__main__":
    main()
