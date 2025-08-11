import os

# 摄像头
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 推理后端：'roboflow' 或 'ultralytics'
# 需求：默认 ultralytics（离线本地权重）
INFERENCE_BACKEND = os.getenv("CARD_BACKEND", "ultralytics").lower()

# Roboflow 模型配置（备用在线推理，需 ROBOFLOW_API_KEY）
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "augmented-startups")  # 参考项目
ROBOFLOW_PROJECT   = os.getenv("ROBOFLOW_PROJECT", "playing-cards-ow27d")
ROBOFLOW_VERSION   = int(os.getenv("ROBOFLOW_VERSION", "4"))
ROBOFLOW_CONF      = float(os.getenv("ROBOFLOW_CONF", "0.5"))

# Ultralytics 本地权重（将 .pt 下载/复制到此路径）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
# https://github.com/CallMeCore/Playing-Cards-Object-Detection/tree/main/final_models/yolov8m_synthetic.pt
ULTRALYTICS_WEIGHTS = os.getenv("ULTRA_WEIGHTS", os.path.join(MODELS_DIR, "playing_cards_yolov8.pt"))
ULTRA_CONF = float(os.getenv("ULTRA_CONF", "0.25"))  # 从0.3进一步降到0.25

# 可视化
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2  # 从0.8增大到1.2，让字体更清楚
THICKNESS = 2
COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (0, 0, 255)
COLOR_HIGHLIGHT = (0, 255, 255)

# 24点解法显示配置
SOLUTION_FONT_SCALE = 1.5  # 解法文本用更大字体
SOLUTION_THICKNESS = 3     # 解法文本用更粗线条

# 分行聚合：用目标框高度做相对阈值
ROW_MAX_DELTA_RATIO = 0.5  # 同一行中y中心差不超过"平均框高 * 此比例"

os.makedirs(MODELS_DIR, exist_ok=True)