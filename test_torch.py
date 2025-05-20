import torch
from torchvision import models, transforms
#from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import ResNet18_Weights, resnet18
from PIL import Image
import requests
from io import BytesIO
import time

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

#img_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
#response = requests.get(img_url,verify=False)
#img = Image.open(BytesIO(response.content)).convert("RGB")
img = Image.open('Pug_600.jpg').convert("RGB")

# 3. 定義前處理流程（與模型訓練時一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 轉為 Tensor 並自動除以 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 增加 batch 維度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
input_batch = input_batch.to(device)

print(input_batch)

start_time =  time.time()
with torch.no_grad():
    output = model(input_batch)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"執行時間：{elapsed_time:.2f} 秒")

print("========== output ============")
print(output)
print(type(output))

probabilities = torch.nn.functional.softmax(output[0], dim=0)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL,verify=False).text.strip().split("\n")

top5 = torch.topk(probabilities, 5)
print("=======  torch.topk(probabilities, 5) =============")
print(top5)
print(type(top5))


for i in range(5):
    idx = top5.indices[i].item()
    prob = top5.values[i].item()
    print(f"{labels[idx]}: {prob:.4f}")

