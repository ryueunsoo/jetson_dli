# jetson_dli
 1. Jetson Nano Setting 준비물
  
        - jetson nano 4gb


  
        - c type power adapter
  
        - 와이파이 동글
  
        - 웹캠(USB Camera), 또는 CSI Camera (라즈베리파이 V2)
  
        - 64기가 이상 마이크로sd카드
  
        - 그외 쿨링펜, lcd, 또는 모니터. hdmi
    <b> jetson image
    ![화면 캡처 2024-09-25 160828](https://github.com/user-attachments/assets/ab335c3d-bd93-4ee0-a9bc-9c3522276e1f)
    ![image](https://github.com/user-attachments/assets/ca75b378-5444-4ae2-8031-3fd46992492a)

<b> 2. jetson nano에 대하여

<b>  welcome 부터 따라하기
       https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-RX-02+V2&unit=block-v1:DLI+S-RX-02+V2+type@vertical+block@aba5104413ae454c8c63a6f301925337

<b> 3. jetpack downloads 

<b>      https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write

<b> 4. 이미지  굽기 위해 필요한 것들.

       4-1. sd card formatter ---> download
       4-2. balenaetcher download --->  이미지 굽기
       4-3. 제슨나노에 sd넣고 우분투 설치

  #####     
<b> 5. 쿨링팬 설치
``` bash
sudo sh -c 'echo 128 max 255 > /sys/devices/pwm-fan/target_pwm'
```
<b>  6. usb-camera  얼굴의 코 눈 인식하는 것도 해봄, 이미지 캡쳐와 영상 녹화 cctv기능 구현 j는 이미지 캡쳐, 1은 영상 녹화 시작 0은 영상녹화 스톱
```
git clone https://github.com/jetsonhacks/USB-Camera.git
cd USB-Camera
ls
python3 usb-camera-gst.py 
python3  face-detect-usb.py
nvgstcapture-1.0 --mode=1 --camsrc=0 --cap-dev-node=0
j  
nvgstcapture-1.0 --mode=2 --camsrc=0 --cap-dev-node=0
mode1사진 mode2영상
1
0
```
  <img width="80%" src="https://github.com/jetsonmom/dli/issues/2#issue-2560812440"/>



<b>  7. 한글 설치 , reboot 한 후 오른쪽 하단 키보드 모양을 오른쪽 마우스 클릭→ configure click
```
참고 링크 https://driz2le.tistory.com/253
```
```
sudo apt-get update
sudo apt-get install fcitx-hangul
reboot
```
       
<b> 8. 제슨 알아보고 설치하기
  
  [https://developer.nvidia.com/embedded/learn/jetson-nano-2gb-devkit-user-guide#id-.JetsonNano2GBDeveloperKitUserGuidevbatuu_v1.0-DeveloperKitSetup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)
<b> 9. image classification 준비
```
dli@dli-desktop:~$ mkdir -p ~/nvdli-data
dli@dli-desktop:~$ sudo docker run --runtime nvidia -it --rm --network host \
>     --memory=500M --memory-swap=4G \
>     --volume ~/nvdli-data:/nvdli-nano/data \
>     --volume /tmp/argus_socket:/tmp/argus_socket \
>     --device /dev/video0 \
>     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.2-r32.7.1kr
```
<b>  결과 


![image](https://github.com/user-attachments/assets/634eaeeb-1a8f-4bff-a953-55663eef1c7e)

카메라 없어서 생기는 에러로 카메라 연결하고 다시 명령한다
dli@dli-desktop:~$ sudo docker run --runtime nvidia -it --rm --network host \
>     --memory=500M --memory-swap=4G \
>     --volume ~/nvdli-data:/nvdli-nano/data \
>     --volume /tmp/argus_socket:/tmp/argus_socket \
>     --device /dev/video0 \
>     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.2-r32.7.1kr

<b> 결과에 다음과 같은 글이 써진다.
allow 10 sec for JupyterLab to start @ http://192.168.0.152:8888 (password dlinano)
JupterLab logging location:  /var/log/jupyter.log  (inside the container)
root@dli-desktop:/nvdli-nano# 
웹브라우저를 열고 192.168.0.152:8888 를 친다

<b> 결과 

![image](https://github.com/user-attachments/assets/f6f89e60-8b49-478e-ae58-c2238c694679)
![image](https://github.com/user-attachments/assets/645f81ca-8fb1-49b9-ae38-d118c5e07eb3)

http://192.168.0.152:8888/lab/tree/classification/classification_interactive.ipynb

<b> swap메모리가 적으면 thumup 프로젝트 할 때 동영상이 나오지 않고 사진으로 되어 데이터 수집을 할 수 없다. 그래서 미리 스왑을 해준다
```
sudo systemctl disable nvzramconfig

sudo systemctl set-default multi-user.target

sudo fallocate -l 10G /mnt/10GB.swap
sudo chmod 600 /mnt/10GB.swap
sudo mkswap /mnt/10GB.swap

sudo su
echo "/mnt/10GB.swap swap swap defaults 0 0" >> /etc/fstab
exit

sudo reboot
```

<b> 시스템을 GUI 모드로 설정:

```
sudo systemctl set-default graphical.target

reboot

```

<b> Camera
먼저 카메라를 생성하고 running으로 설정합니다. 사용 중인 카메라 유형(USB 또는 CSI)에 따라 적절한 카메라 선택 라인을 주석 해제합니다. 이 셀을 실행하는 데 몇 초가 걸릴 수 있습니다. jupyterlab에서 실행


<b> 10. image classification  -  Thumbs Project  using ResNet

```
# Check device number
!ls -ltrh /dev/video*
```
```
from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera

# for USB Camera (Logitech C270 webcam), uncomment the following line
camera = USBCamera(width=224, height=224, capture_device=0) # confirm the capture_device number

# for CSI Camera (Raspberry Pi Camera Module V2), uncomment the following line
# camera = CSICamera(width=224, height=224, capture_device=0) # confirm the capture_device number

camera.running = True
print("camera created")
```
<b>  Task
그런 다음 프로젝트 작업 TASK과 수집할 데이터 범주 CATEGORIES를 정의합니다. 선택한 이름으로 여러 데이터세트 DATASETS에 대한 공간을 정의할 수도 있습니다.

작성 중인 분류 작업에 대해 연결된 줄을 주석 해제/수정하고 실행합니다. 이 셀을 실행하는 데 몇 초밖에 걸리지 않습니다
```
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

TASK = 'thumbs'
# TASK = 'emotions'
# TASK = 'fingers'
# TASK = 'diy'

CATEGORIES = ['thumbs_up', 'thumbs_down']
# CATEGORIES = ['none', 'happy', 'sad', 'angry']
# CATEGORIES = ['1', '2', '3', '4', '5']
# CATEGORIES = [ 'diy_1', 'diy_2', 'diy_3']

DATASETS = ['A', 'B']
# DATASETS = ['A', 'B', 'C']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset('../data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)
    
print("{} task with {} categories defined".format(TASK, CATEGORIES))
```
```
# Set up the data directory location if not there already
DATA_DIR = '/nvdli-nano/data/classification/'
!mkdir -p {DATA_DIR}
```
<b>  Data Collection
아래 셀을 실행하여 데이터 수집 도구 위젯을 만든다
```
import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

# initialize active dataset
dataset = datasets[DATASETS[0]]

# unobserve all callbacks from camera in case we are running this cell for second time
camera.unobserve_all()

# create image preview
camera_widget = ipywidgets.Image()
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

# create widgets
dataset_widget = ipywidgets.Dropdown(options=DATASETS, description='dataset')
category_widget = ipywidgets.Dropdown(options=dataset.categories, description='category')
count_widget = ipywidgets.IntText(description='count')
save_widget = ipywidgets.Button(description='add')

# manually update counts at initialization
count_widget.value = dataset.get_count(category_widget.value)

# sets the active dataset
def set_dataset(change):
    global dataset
    dataset = datasets[change['new']]
    count_widget.value = dataset.get_count(category_widget.value)
dataset_widget.observe(set_dataset, names='value')

# update counts when we select a new category
def update_counts(change):
    count_widget.value = dataset.get_count(change['new'])
category_widget.observe(update_counts, names='value')

# save image for category and update counts
def save(c):
    dataset.save_entry(camera.value, category_widget.value)
    count_widget.value = dataset.get_count(category_widget.value)
save_widget.on_click(save)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget]), dataset_widget, category_widget, count_widget, save_widget
])

# display(data_collection_widget)
print("data_collection_widget created")
```
<b> Model
다음 셀을 실행하여 뉴럴 네트워크를 정의하고 프로젝트에 필요한 출력과 일치하도록 완전히 연결된 레이어(fc)를 조정합니다
```
import torch
import torchvision


device = torch.device('cuda')

# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, len(dataset.categories))

# SQUEEZENET 
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, len(dataset.categories), kernel_size=1)
# model.num_classes = len(dataset.categories)

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))

# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))
    
model = model.to(device)

model_save_button = ipywidgets.Button(description='save model')
model_load_button = ipywidgets.Button(description='load model')
model_path_widget = ipywidgets.Text(description='model path', value='/nvdli-nano/data/classification/my_model.pth')

def load_model(c):
    model.load_state_dict(torch.load(model_path_widget.value))
model_load_button.on_click(load_model)
    
def save_model(c):
    torch.save(model.state_dict(), model_path_widget.value)
model_save_button.on_click(save_model)

model_widget = ipywidgets.VBox([
    model_path_widget,
    ipywidgets.HBox([model_load_button, model_save_button])
])

# display(model_widget)
print("model configured and model_widget created")
```
<b> Live Execution
아래 셀을 실행하여 실시간 실행 위젯을 설정

```

import threading
import time
from utils import preprocess
import torch.nn.functional as F

state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='state', value='stop')
prediction_widget = ipywidgets.Text(description='prediction')
score_widgets = []
for category in dataset.categories:
    score_widget = ipywidgets.FloatSlider(min=0.0, max=1.0, description=category, orientation='vertical')
    score_widgets.append(score_widget)

def live(state_widget, model, camera, prediction_widget, score_widget):
    global dataset
    while state_widget.value == 'live':
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        prediction_widget.value = dataset.categories[category_index]
        for i, score in enumerate(list(output)):
            score_widgets[i].value = score
            
def start_live(change):
    if change['new'] == 'live':
        execute_thread = threading.Thread(target=live, args=(state_widget, model, camera, prediction_widget, score_widget))
        execute_thread.start()

state_widget.observe(start_live, names='value')

live_execution_widget = ipywidgets.VBox([
    ipywidgets.HBox(score_widgets),
    prediction_widget,
    state_widget
])

# display(live_execution_widget)
print("live_execution_widget created")
```
<b>  Training and Evaluation¶
다음 셀을 실행하여 트레이너를 정의하고 위젯을 실행하여 트레이너를 제어합니다

```
BATCH_SIZE = 8

optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

epochs_widget = ipywidgets.IntText(description='epochs', value=1)
eval_button = ipywidgets.Button(description='evaluate')
train_button = ipywidgets.Button(description='train')
loss_widget = ipywidgets.FloatText(description='loss')
accuracy_widget = ipywidgets.FloatText(description='accuracy')
progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description='progress')

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, eval_button, train_button, accuracy_widget, loss_widget, progress_widget, state_widget
    
    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        state_widget.value = 'stop'
        train_button.disabled = True
        eval_button.disabled = True
        time.sleep(1)

        if is_training:
            model = model.train()
        else:
            model = model.eval()
        while epochs_widget.value > 0:
            i = 0
            sum_loss = 0.0
            error_count = 0.0
            for images, labels in iter(train_loader):
                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                if is_training:
                    # zero gradients of parameters
                    optimizer.zero_grad()

                # execute model to get outputs
                outputs = model(images)

                # compute loss
                loss = F.cross_entropy(outputs, labels)

                if is_training:
                    # run backpropogation to accumulate gradients
                    loss.backward()

                    # step optimizer to adjust parameters
                    optimizer.step()

                # increment progress
                error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
                count = len(labels.flatten())
                i += count
                sum_loss += float(loss)
                progress_widget.value = i / len(dataset)
                loss_widget.value = sum_loss / i
                accuracy_widget.value = 1.0 - error_count / i
                
            if is_training:
                epochs_widget.value = epochs_widget.value - 1
            else:
                break
    except e:
        pass
    model = model.eval()

    train_button.disabled = False
    eval_button.disabled = False
    state_widget.value = 'live'
    
train_button.on_click(lambda c: train_eval(is_training=True))
eval_button.on_click(lambda c: train_eval(is_training=False))
    
train_eval_widget = ipywidgets.VBox([
    epochs_widget,
    progress_widget,
    loss_widget,
    accuracy_widget,
    ipywidgets.HBox([train_button, eval_button])
])

# display(train_eval_widget)
print("trainer configured and train_eval_widget created")
```
<b>  
Display the Interactive Tool!

```
# Combine all the widgets into one display
all_widget = ipywidgets.VBox([
    ipywidgets.HBox([data_collection_widget, live_execution_widget]), 
    train_eval_widget,
    model_widget
])

display(all_widget)
```
<b>  Before you go...

카메라 및/또는 노트북 커널을 종료하여 카메라 리소스를 해제합니다.

```
# Attention!  Execute this cell before moving to another notebook
# The USB camera application only requires that the notebook be reset
# The CSI camera application requires that the 'camera' object be specifically released

import os
import IPython

if type(camera) is CSICamera:
    print("Ignore 'Exception in thread' tracebacks\n")
    camera.cap.release()

os._exit(00)
```

<b>  
  https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-RX-02+V2&unit=block-v1:DLI+S-RX-02+V2+type@vertical+block@aabe204272214ba69309581d388b0734


  

  <b> 11. image regression  -  Face XY Project

  https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-RX-02+V2&unit=block-v1:DLI+S-RX-02+V2+type@vertical+block@76a2873eb69946b4928c4f8432e04314
