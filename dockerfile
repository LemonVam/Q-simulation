FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04


WORKDIR /pj

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install jupyter
RUN pip install --upgrade pip

RUN pip install numpy pandas matplotlib scipy scikit-learn requests Django jsonschema cirq tensorflow

RUN pip install torch torchvision

COPY . .

CMD ["tail", "-f", "/dev/null"]




# 使用带有 CUDA 12.6.2 和 cuDNN 的 NVIDIA 基础镜像
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# 更新并安装 Python 和 pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 检查并创建 python 和 pip 的符号链接
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 升级 pip 并安装所需的 Python 包
RUN pip install --upgrade pip
RUN pip install jupyter numpy pandas matplotlib scipy scikit-learn requests Django jsonschema cirq tensorflow gensim torch torchvision

# 设置工作目录
WORKDIR /pj

# 复制当前目录内容到工作目录
COPY . .

# 保持容器持续运行
CMD ["bash"]
