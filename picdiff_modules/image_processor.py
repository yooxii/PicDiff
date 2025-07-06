import cv2 as cv
import numpy as np
from typing import Optional

class ImageProcessor:
    def __init__(self):
        """初始化图像处理器"""
        self.adaptive_params = {  # 自适应处理参数
            'block_size': 45,    # 块大小
            'c': 7,              # 常数C
            'threshold': 0       # 阈值
        }
    
    def analyze_image(self, img: np.ndarray) -> dict:
        """分析图像特征用于自适应处理
        
        参数:
            img: 输入图像
        返回:
            包含亮度、对比度和锐度特征的字典
        """
        # 确保图像是3通道BGR格式
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图
        mean_val = np.mean(gray)    # 计算平均亮度
        std_val = np.std(gray)      # 计算对比度(标准差)
        blur = cv.Laplacian(gray, cv.CV_64F).var()  # 计算锐度(拉普拉斯方差)
        
        return {
            'brightness': mean_val,  # 亮度
            'contrast': std_val,     # 对比度
            'sharpness': blur        # 锐度
        }
    
    def adjust_params(self, features: dict):
        """根据图像特征调整处理参数
        
        参数:
            features: 图像特征字典
        """
        # 调整自适应阈值参数
        if features['contrast'] < 50:  # 低对比度图像
            self.adaptive_params['block_size'] = 35
            self.adaptive_params['c'] = 10
        elif features['contrast'] > 100:  # 高对比度图像
            self.adaptive_params['block_size'] = 55
            self.adaptive_params['c'] = 5
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """自适应图像预处理
        
        参数:
            img: 输入图像
        返回:
            预处理后的二值图像
        """
        features = self.analyze_image(img)  # 分析图像特征
        self.adjust_params(features)        # 调整参数
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度
        
        # 自适应阈值处理
        binary = cv.adaptiveThreshold(
            gray, 255, 
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权
            cv.THRESH_BINARY,                # 二值化
            self.adaptive_params['block_size'],  # 块大小
            self.adaptive_params['c']           # 常数C
        )
        
        return binary

    def detect_product_region(self, img: np.ndarray) -> Optional[tuple]:
        """检测图像中的产品区域
        
        参数:
            img: 输入图像
        返回:
            产品区域坐标(x,y,w,h)或None
        """
        # 转换为灰度图
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # 自适应阈值处理
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 51, 5
        )
        
        # 形态学操作去除小噪点
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        cleaned = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 获取最大轮廓(假设是产品)
        largest_cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_cnt)
        
        # 检查产品区域是否合理(至少占图像面积的10%)
        if w * h < img.shape[0] * img.shape[1] * 0.1:
            return None
            
        return (x, y, w, h)

    def resize_to_standard(self, img: np.ndarray, width: int = 2000) -> np.ndarray:
        """将图像调整为标准尺寸(先裁剪背景再缩放)
        
        参数:
            img: 输入图像
            width: 目标宽度(默认2000)
        返回:
            调整大小后的图像
        异常:
            如果无法检测到产品区域，抛出ValueError
        """
        # 检测产品区域
        product_region = self.detect_product_region(img)
        
        if product_region is None:
            raise ValueError("无法检测到产品区域，请上传背景对比更明显的图片")
            
        x, y, w, h = product_region
        
        # 裁剪产品区域
        cropped = img[y:y+h, x:x+w]
        
        # 缩放至标准尺寸
        h, w = cropped.shape[:2]
        ratio = width / w
        dim = (width, int(h * ratio))
        
        return cv.resize(cropped, dim, interpolation=cv.INTER_AREA)
