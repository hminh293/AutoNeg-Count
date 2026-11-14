"""AutoNeg-Count — giảm đếm nhầm trong zero-shot counting trên nền CountGD++.

Các module không cần GPU (cluster, cpe, anm, threshold, metrics, datasets) không được
import torch ở cấp module, để test chạy được trên máy không GPU.
"""
__version__ = "0.1.0"
