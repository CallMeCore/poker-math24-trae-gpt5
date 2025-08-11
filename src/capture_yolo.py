import cv2
import numpy as np
import time
from typing import List, Dict
import re
import itertools
from fractions import Fraction
from collections import deque
import pyttsx3
import threading
import queue
from PIL import Image, ImageDraw, ImageFont

try:
    import config
    from yolo_engine import YoloDetector
except ImportError:
    # 万一从其他工作目录运行，尝试把当前脚本目录加入 sys.path
    import sys, os
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    import config
    from yolo_engine import YoloDetector


def _group_by_rows(dets):
    if not dets:
        return []
    centers = []
    heights = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((cx, cy))
        heights.append(y2 - y1)
    avg_h = np.mean(heights) if heights else 1.0
    rows = []
    used = [False] * len(dets)
    for i in range(len(dets)):
        if used[i]:
            continue
        base_cy = centers[i][1]
        row = [i]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            if abs(centers[j][1] - base_cy) <= avg_h * config.ROW_MAX_DELTA_RATIO:
                row.append(j)
                used[j] = True
        row.sort(key=lambda k: centers[k][0])
        rows.append(row)
    rows.sort(key=lambda idxs: np.mean([centers[k][1] for k in idxs]))
    return rows, centers


def draw_detections(frame, dets: List[Dict]):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cls_name = d["cls"]
        conf = d["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_BOX, config.THICKNESS)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.THICKNESS)
        y_text = max(y1 - 8, th + 4)
        # 文本背景
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 2, y_text + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 1, y_text - 2), config.FONT, config.FONT_SCALE, config.COLOR_TEXT, config.THICKNESS, cv2.LINE_AA)


def draw_chinese_text(img, text, position, font_size=30, color=(0, 255, 0), thickness=2):
    """
    在OpenCV图像上绘制中文文本
    """
    try:
        # 转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试使用系统中文字体
        try:
            # Windows 中文字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttf", font_size)  # 微软雅黑
            except:
                font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB，OpenCV使用BGR
        
        # 转换回OpenCV格式
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img[:] = img_cv[:]
        
    except Exception as e:
        # 如果中文字体失败，回退到英文显示
        print(f"中文字体显示失败，使用英文: {e}")
        # 将中文转为拼音或英文
        text_en = text.replace("等待", "Wait").replace("后识别下一组", "s for next").replace("未找到24点解法", "No 24-point solution")
        cv2.putText(img, text_en, position, config.FONT, config.FONT_SCALE, color, thickness, cv2.LINE_AA)


def main():
    det = YoloDetector()  # 会根据 config.INFERENCE_BACKEND 自动选择 ultralytics/roboflow
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("无法打开摄像头，请检查 CAMERA_INDEX")
        return

    fps_ts = time.time()
    frame_count = 0

    # 立即求解 + 冷却（取消帧稳定计数）
    last_tuple = None
    cooldown_until = 0.0
    last_solution_text = None
    # 替换为异步TTS
    tts = TTSWorker(rate=180)
    # 新增：多帧聚合窗口和历史集合（基于牌(点数+花色)）
    agg_N = 2
    frames_cards_sets = []
    last_seen_conf = {}  # 记录每张牌(点数+花色)最近一次的置信度

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头失败")
            break

        now = time.time()
        dets = det.predict(frame)
        draw_detections(frame, dets)

        # 从检测中提取“牌(点数+花色)”及置信度，去重取每张牌最高置信度
        curr_cards_conf = {}  # {(rank,suit): conf}
        for d in dets:
            parsed = parse_card_from_cls_name(d["cls"])
            if parsed is None:
                continue
            rank, suit = parsed
            if not (1 <= rank <= 13):
                continue
            conf = float(d["conf"])
            if (rank, suit) not in curr_cards_conf or conf > curr_cards_conf[(rank, suit)]:
                curr_cards_conf[(rank, suit)] = conf

        curr_cards = list(curr_cards_conf.keys())

        # 冷却期提示
        if now < cooldown_until:
            remain = int(cooldown_until - now)
            tip = f"等待 {remain}s 后识别下一组"
            # 使用中文字体绘制
            draw_chinese_text(frame, tip, (10, 90), font_size=35, color=(0, 255, 255))
            
            if last_solution_text:
                # 解法用更大更清楚的字体
                draw_chinese_text(frame, last_solution_text, (10, 140), font_size=40, color=(0, 255, 0))
        else:
            # 非冷却期：维护最近N帧集合（以牌为单位）
            frames_cards_sets.append(set(curr_cards))
            while len(frames_cards_sets) > agg_N:
                frames_cards_sets.pop(0)

            # 更新最近置信度
            for k, v in curr_cards_conf.items():
                last_seen_conf[k] = v

            # 显示聚合N
            draw_chinese_text(frame, f"聚合N = {agg_N}", (10, 60), font_size=32, color=(255, 255, 0))

            # 候选：优先当前帧已凑齐4张；否则用最近N帧并集去重后凑齐4张
            candidate_cards = None
            if len(curr_cards) >= 4:
                # 取当前帧中按置信度最高的4张牌
                candidate_cards = [k for k, _ in sorted(curr_cards_conf.items(), key=lambda kv: kv[1], reverse=True)[:4]]
            else:
                # 最近N帧并集（以牌为单位）
                union_cards = set()
                for sset in frames_cards_sets:
                    union_cards |= sset
                if len(union_cards) >= 4:
                    # 用最近置信度排序，取前4张
                    candidate_cards = sorted(list(union_cards), key=lambda k: last_seen_conf.get(k, 0.0), reverse=True)[:4]

            if candidate_cards:
                # 求解时只用点数，但保证是4张牌（允许相同点数不同花色）
                ranks_for_solve = [r for (r, s) in candidate_cards]
                solution = solve_24_prefer_simple(ranks_for_solve)

                cards_label = ", ".join(format_card_key(c) for c in candidate_cards)
                if solution:
                    last_solution_text = f"{cards_label} → {solution} = 24"
                    say_text = expr_to_speech(solution)
                else:
                    last_solution_text = f"{cards_label} → 未找到24点解法"
                    say_text = "没有找到二十四点的解法。"

                # 异步播报（不阻塞画面）
                tts.speak(say_text)

                # 进入8秒冷却
                cooldown_until = time.time() + 8
                last_tuple = tuple(candidate_cards)
                # 清空历史，准备下一组
                frames_cards_sets.clear()

        # 粗略 FPS - 用更大字体
        frame_count += 1
        if frame_count >= 10:
            now2 = time.time()
            fps = frame_count / (now2 - fps_ts + 1e-6)
            fps_ts = now2
            frame_count = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40), config.FONT, config.FONT_SCALE, config.COLOR_HIGHLIGHT, config.THICKNESS, cv2.LINE_AA)

        cv2.imshow("Poker YOLO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        # 调整聚合窗口大小（1..5）
        if key == ord('+'):
            agg_N = min(5, agg_N + 1)
        elif key == ord('-'):
            agg_N = max(1, agg_N - 1)

    cap.release()
    cv2.destroyAllWindows()


ACE_WORDS = {"ace", "a"}
TEN_WORDS = {"ten", "t", "10"}
JACK_WORDS = {"jack", "j", "11"}
QUEEN_WORDS = {"queen", "q", "12"}
KING_WORDS = {"king", "k", "13"}

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def parse_rank_from_cls_name(cls: str):
    if not cls:
        return None
    s = str(cls).strip().lower()
    
    # 首先专门处理 "10" 的情况，避免被误解析
    if "10" in s:
        return 10
    
    # 再找其他数字（2-9，排除1避免和10冲突）
    m = re.search(r'\b([2-9])\b', s)
    if m:
        return int(m.group(1))
    
    # 处理A（1）的情况
    if re.search(r'\b1\b', s) or any(t in ACE_WORDS for t in re.split(r'[^a-z0-9]+', s)):
        return 1
    
    # 再查字母缩写和英文单词
    tokens = re.split(r'[^a-z0-9]+', s)
    tokens = [t for t in tokens if t]
    
    if any(t in TEN_WORDS for t in tokens) or "ten" in tokens:
        return 10
    if any(t in JACK_WORDS for t in tokens):
        return 11
    if any(t in QUEEN_WORDS for t in tokens):
        return 12
    if any(t in KING_WORDS for t in tokens):
        return 13
    
    # 英文数字
    for w, v in NUM_WORDS.items():
        if w in tokens:
            return v
    
    # 常见52类压缩名如 "ah","qs","td" 等：首字符是 a23456789tjqk
    m2 = re.match(r'\b([a2-9tjqk])', s)
    if m2:
        ch = m2.group(1)
        mapping = {'a':1,'t':10,'j':11,'q':12,'k':13}
        if ch.isdigit():
            return int(ch)
        return mapping.get(ch, None)
    
    return None

def parse_card_from_cls_name(cls: str):
    """
    返回 (rank:int, suit:str) ，suit in {'h','d','c','s'}；解析失败返回 None
    """
    if not cls:
        return None
    s = str(cls).strip().lower()

    # 先解析点数（复用已有的点数解析）
    rank = parse_rank_from_cls_name(s)
    if rank is None:
        return None

    suit = None
    # 1) 词汇匹配
    if "heart" in s:
        suit = 'h'
    elif "diamond" in s:
        suit = 'd'
    elif "club" in s:
        suit = 'c'
    elif "spade" in s:
        suit = 's'

    # 2) 压缩写法匹配，支持 "10h", "qs", "4d", "qc" 等格式
    if suit is None:
        # 先匹配 10 + 花色的情况
        m = re.search(r'10([hdcs])', s)
        if m:
            suit = m.group(1)
        else:
            # 再匹配单字符 + 花色的情况
            m = re.search(r'([atjqk2-9])([hdcs])', s)
            if m:
                suit = m.group(2)

    # 3) 单字母花色（较弱，避免误判）
    if suit is None:
        tokens = re.split(r'[^a-z0-9]+', s)
        tokens = [t for t in tokens if t]
        for t in tokens:
            if t in ('h', 'd', 'c', 's'):
                suit = t
                break

    if suit is None:
        return None
    return (rank, suit)

def format_card_key(card):
    """
    (rank:int, suit:str) -> 'Ah','Td','7c' 等
    """
    rank, suit = card
    rmap = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
    rstr = rmap.get(rank, str(rank))
    return f"{rstr}{suit}"

# 组合两个表达式节点
def combine(a, b):
    # a,b: (val(Fraction), expr(str), meta dict)
    results = []
    va, ea, ma = a
    vb, eb, mb = b

    def make(val, expr, op, int_pref):
        # 记录是否出现负数（包含子树历史或本次结果为负）
        new_neg = ma.get("neg", False) or mb.get("neg", False) or (val < 0)
        # 评分：更偏好 + 和 *，不喜欢 / 和 -；更偏好整数中间结果；中间出现负数给予额外惩罚
        score = ma["score"] + mb["score"]
        if op in ('+', '*'):
            score += 1
        elif op in ('-', '/'):
            score += 3
        if int_pref:
            score -= 1
        else:
            score += 2
        if val < 0:
            score += 10  # 本步为负，强惩罚
        return (val, expr, {"score": score, "neg": new_neg})

    # a+b, a-b, b-a, a*b, a/b, b/a
    results.append(make(va + vb, f"({ea}+{eb})", '+', (va + vb).denominator == 1))
    results.append(make(va - vb, f"({ea}-{eb})", '-', (va - vb).denominator == 1))
    results.append(make(vb - va, f"({eb}-{ea})", '-', (vb - va).denominator == 1))
    results.append(make(va * vb, f"({ea}*{eb})", '*', (va * vb).denominator == 1))
    if vb != 0:
        results.append(make(va / vb, f"({ea}/{eb})", '/', (va / vb).denominator == 1))
    if va != 0:
        results.append(make(vb / va, f"({eb}/{ea})", '/', (vb / va).denominator == 1))
    return results

def solve_24_prefer_simple(nums):
    # nums: list[int] 长度4
    targets = {Fraction(24,1)}
    best = None  # (expr, score, has_negative)
    # 对所有排列做搜索
    for perm in set(itertools.permutations(nums, 4)):
        items = [(Fraction(x,1), str(x), {"score": 0, "neg": False}) for x in perm]
        # 递归合并
        def search(arr):
            nonlocal best
            # 若已经找到“无负数且分数<=0”的最优形态，可做简单剪枝
            if best and (best[2] is False) and best[1] <= 0:
                return
            if len(arr) == 1:
                val, expr, meta = arr[0]
                if val in targets:
                    cand = (expr, meta["score"], meta.get("neg", False))
                    if best is None or (cand[2], cand[1]) < (best[2], best[1]):
                        # 先比较是否出现负数，其次比较评分
                        best = cand
                return
            n = len(arr)
            for i in range(n):
                for j in range(i+1, n):
                    rest = [arr[k] for k in range(n) if k != i and k != j]
                    for c in combine(arr[i], arr[j]):
                        search(rest + [c])
        search(items)
    if best:
        expr = best[0]
        return expr
    return None

class TTSWorker:
    def __init__(self, rate=180):
        self.q = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._rate = rate
        self._thread.start()

    def _run(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                # 每次都重新初始化引擎，避免状态异常
                engine = pyttsx3.init()
                engine.setProperty('rate', self._rate)
                engine.say(text)
                engine.runAndWait()
                # 显式删除引擎对象
                del engine
                print(f"[TTS播报完成] {text}")
            except Exception as e:
                print(f"TTS播报失败: {e}")

    def speak(self, text: str):
        try:
            # 最新任务优先：队列满则清空旧任务
            while self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put_nowait(text)
            print(f"[TTS入队] {text}")  # 控制台留痕，便于确认确实触发了播报
        except Exception as e:
            print(f"TTS入队失败: {e}")

def expr_to_speech(expr: str) -> str:
    # 去掉括号，符号替换为中文读法  
    s = expr.replace('(', '').replace(')', '')
    s = s.replace('*', ' 乘以 ').replace('+', ' 加 ').replace('-', ' 减 ').replace('/', ' 除以 ')
    # 多个空格压缩
    s = re.sub(r'\s+', ' ', s).strip()
    return f"二十四点解法：{s}。"
if __name__ == "__main__":
    main()