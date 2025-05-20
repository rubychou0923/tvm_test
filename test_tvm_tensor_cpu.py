# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
.. _optimize_model:

End-to-End Optimize Model
=========================
This tutorial demonstrates how to optimize a machine learning model using Apache TVM. We will
use a pre-trained ResNet-18 model from PyTorch and end-to-end optimize it using TVM's Relax API.
Please note that default end-to-end optimization may not suit complex models.
"""

######################################################################
# Preparation
# -----------
# First, we prepare the model and input information. We use a pre-trained ResNet-18 model from
# PyTorch.

import os
import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision import transforms
import time
from PIL import Image
import requests
from io import BytesIO
import time


######################################################################
# Review Overall Flow
# -------------------
# .. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_overall_flow.svg
#    :align: center
#    :width: 80%
#
# The overall flow consists of the following steps:
#
# - **Construct or Import a Model**: Construct a neural network model or import a pre-trained
#   model from other frameworks (e.g. PyTorch, ONNX), and create the TVM IRModule, which contains
#   all the information needed for compilation, including high-level Relax functions for
#   computational graph, and low-level TensorIR functions for tensor program.
# - **Perform Composable Optimizations**: Perform a series of optimization transformations,
#   such as graph optimizations, tensor program optimizations, and library dispatching.
# - **Build and Universal Deployment**: Build the optimized model to a deployable module to the
#   universal runtime, and execute it on different devices, such as CPU, GPU, or other accelerators.
#


######################################################################
# Convert the model to IRModule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next step, we convert the model to an IRModule using the Relax frontend for PyTorch for further
# optimization.

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
# Give an example argument to torch.export
example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# Skip running in CI environment
IS_IN_CI = os.getenv("CI", "") == "true"

if not IS_IN_CI:
    # Convert the model to IRModule
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True)

    mod, params = relax.frontend.detach_params(mod)
    mod.show()

######################################################################
# IRModule Optimization
# ---------------------
# Apache TVM Unity provides a flexible way to optimize the IRModule. Everything centered
# around IRModule optimization can be composed with existing pipelines. Note that each
# transformation can be combined as an optimization pipeline via ``tvm.ir.transform.Sequential``.
#
# In this tutorial, we focus on the end-to-end optimization of the model via auto-tuning. We
# leverage MetaSchedule to tune the model and store the tuning logs to the database. We also
# apply the database to the model to get the best performance.
#

target_str="llvm"
mod = relax.get_pipeline("zero")(mod)
mod.show()

mod = mod
print("============ mod show() =============")
mod["main"].show()

print("========== end of mod show() =============")


#img_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
#response = requests.get(img_url,verify=False)
#img = Image.open(BytesIO(response.content)).convert("RGB")
img = Image.open('Pug_600.jpg').convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 轉為 Tensor 並自動除以 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 增加 batch 維度

times = 1
if not IS_IN_CI:
    ex = tvm.compile(mod, target=target_str)
    dev = tvm.cpu()
    vm = relax.VirtualMachine(ex, dev)


    print("input_batch shape:", input_batch.shape)  # 應該是 (1, 3, 224, 224)
    data = input_batch.numpy().astype("float32")
    array=tvm.nd.array(data, dev)
    print(array)
    #gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

    param_exprs = mod["main"].params[1:]  # 第 0 個是輸入 x，跳過
    param_values = params["main"]         # 是 list
    gpu_params = [tvm.nd.array(val, dev) for _, val in zip(param_exprs, param_values)]
    print(gpu_params)

    start_time =  time.time()
    for i in range(times):
        gpu_out = vm["main"](array, *gpu_params).numpy()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"執行時間：{elapsed_time:.2f} 秒")

print("推論輸出 shape:", gpu_out.shape)
torch_tensor = torch.from_numpy(gpu_out)
print(torch_tensor)


probabilities = torch.nn.functional.softmax(torch_tensor[0], dim=0)

# 7. 載入 ImageNet 類別標籤
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL,verify=False).text.strip().split("\n")

# 8. 印出前 5 名結果
top5 = torch.topk(probabilities, 5)
print("======== torch.topk(probabilities, 5) ================")
print(top5)
print(type(top5))

for i in range(5):
    idx = top5.indices[i].item()
    prob = top5.values[i].item()
    print(f"{labels[idx]}: {prob:.4f}")

