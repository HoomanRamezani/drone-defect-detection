# drone-defect-detection
This repo has code related to the drone-defect-detection Project repository. The project's goal is to develop an embedded video frame-classification model that can accurately classify surface defects. The model is lightweight and capable of running on edge devices on-device. To overcome the challenge of limited real-world data, a photo-realistic simulation using Unreal Engine and harnessed OpenCV for synthetic data generation. 

Model detects surface defects with accuracy of 96%, well balanced for sensitivity / precision. Its architecture is now expanded to include a convolutional-RNN to capture temporal information between video frames. It features YOLO object detection, LSTM, automated hyperparameter tuning, and GPU optimization.

# Project Setup
To get started with this project, you will need the following:
```
Python 3.7 or higher
TensorFlow 2.x
OpenCV Library
Unreal Engine (for simulation)
Access to high-performance computing resources for training (preferably with a CUDA-compatible GPU)
```

Clone the repository using:
```
git clone https://github.com/HoomanRamezani/drone-defect-detection
cd drone-defect-detection
drone-defect-detection
```

Install the required Python dependencies:
```
pip install -r requirements.txt
```
## Model Training
To train the model:

Set up your dataset by following the instructions in dataset_setup.md.
Configure your training parameters in config.json.

Start the training process using:
```
python main.py --dataset_dir ./data --epochs 50 --batch_size 32 --learning_rate 0.001
```

You can monitor the training progress via TensorBoard:
```
tensorboard --logdir=path/to/log-files
```

## Evaluation
After training, evaluate your model's performance using the evaluate.py script, ensuring that it meets our target accuracy of 96% for image classification and object detection tasks.
```
python evaluate.py --model_dir path/to/saved_model --data_dir path/to/evaluation_data --batch_size 32
```
