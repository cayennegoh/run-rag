# run-rag
# Build a Docker image
Prerequisites: Assuming docker is set up, nvidia drivers installed
>Create a Dockerfile 

```sh
mkdir rag
cd rag
touch Dockerfile
```

>In dockerfile, enter the following:

```sh
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch
RUN git clone https://github.com/cayennegoh/run-rag

RUN pip install git+https://github.com/huggingface/accelerate
RUN pip install huggingface_hub
RUN pip install pillow
RUN pip install torchvision
RUN pip install sentencepiece
RUN pip install bitsandbytes
RUN pip install fairscale fire blobfile einops
RUN pip install transformers==4.48
RUN pip install chromadb
RUN apt update


```
>Build the Dockerfile
```sh
docker build . -t rag
```

```sh
docker run --gpus all -it -v /home/nvidia/Downloads:/pic --name rag rag
```
>Change /home/nvidia/Downloads to your own path


In a separate terminal 
>Create the dataset
```sh
cd /home/nvidia/Downloads
# change to the directory that u mounted
mkdir dataset
cd dataset
```
>Add in the data by uploading required images into the folder

In the Docker container 
```sh
cd run-rag
vim molmo1bpoint.py 
#In line 58, add in captions for the images according to their order 
python3 molmo1bpoint.py
```
![Screenshot 2025-05-22 091856](https://github.com/user-attachments/assets/82346576-50df-4c7a-8dce-b96126c6603a)

>Choose option 1 to ask system to point    
>Option 2 to query the image

![Screenshot 2025-05-23 113653](https://github.com/user-attachments/assets/3ce94026-3f8c-4a97-984c-1d40cefd6878)
