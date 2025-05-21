#!/bin/bash

# 设置日志文件路径
LOG_FILE="./web.log"
export PATH="/opt/conda/bin:$PATH"

# 指定要激活的虚拟环境名称
export CONDA_ENV_NAME="segchange"

# 动态获取主机的 IP 地址
export GRADIO_SERVER_NAME=$(hostname -i | awk '{print $1}')
export GRADIO_SERVER_PORT=7860
export ORIGINS="*"

# 检查日志文件是否存在
if [ -f "$LOG_FILE" ]; then
    # 清空日志文件
    echo -n "" > "$LOG_FILE"
else
    # 创建日志文件
    touch "$LOG_FILE"
fi

# 输出当前使用的 IP 地址到日志
echo "Using HOST_IP: $GRADIO_SERVER_NAME" | tee -a "$LOG_FILE"

# 激活指定的虚拟环境
source /opt/conda/bin/activate $CONDA_ENV_NAME
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
    exit 1
else
    echo "Activated conda environment: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
fi

# 切换到工作目录
echo "切换到工作目录" | tee -a "$LOG_FILE"
# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "$SCRIPT_DIR" | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR" | tee -a "$LOG_FILE"; exit 1; }


# 启动 Gradio 服务并将日志输出到 web.log
/opt/conda/envs/segchange/bin/python app.py >> "$LOG_FILE" 2>&1 &
GRADIO_PID=$!
echo "Gradio service started with PID: $GRADIO_PID" | tee -a "$LOG_FILE"

# 输出自定义的 Port Forwarding URL
CUSTOM_URL="http://172.16.15.10:30110"
echo -e "Port Forwarding Post URL: \e[4m$CUSTOM_URL\e[0m" | tee -a "$LOG_FILE"


# 检测按键输入，按 Esc 键退出
echo "Press 'Esc' key to stop the service..."
while true; do
    read -rsn1 key
    if [[ $key == $'\e' ]]; then
        echo "Stopping Gradio service..."
        kill -9 $GRADIO_PID
        echo "Service stopped."
        break
    fi
done