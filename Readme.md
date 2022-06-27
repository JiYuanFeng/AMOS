## AMOS2022 Second stage submission example
This instruction provides the guidance of how to pack developed algorithms to a docker image. Please direct any questions or concerns about these instructions or the submission process generally to [AMOS grand challenge forum](https://link-url-here.org). Note that this guide draws heavily on Kits21's [submission guidance](https://github.com/neheller/kits21/tree/master/examples/submission), and we are grateful to the project's developers.

### Submission guidance
AMOS22 is a two-phase challenge. In the second stage, the participants will be asked to upload the inference portion of your algorithm in the form of a docker container. The submission takes place by uploading a saved docker image (single file) containing your inference code, and related short paper (please ues miccai paper template) to our email: miccai.amos22@gmail.com.
The structure of the expected submission of task1 is shown:
- TEAMNAME-task1.tar.gz (docker image)
- TEAMNAME-task1.pdf (short paper)

This image will be loaded on the evaluation system and executed on private servers to run inference on the test images. Naturally, these docker images will NOT have internet access, so please ensure everything you need is included in the image you upload.  Containers will be run on server with a single NVIDIA 3090 card and 4 CPUs with 30GB of CPU memory. Make sure your docker works well with this hardware.

On our servers, the containers will be mounted such that two specific folders are available,  ``/input/`` and ``/output/``. The ``/input/`` contains the test data, where merely a bunch of *.nii.gz test images files. Your docker is expected to produce equivalently named segmentation files in the ``/output/``. The structure of the expected input and output folder is shown:

``` bash
  ├── /input/
  │   └── amos_000x.nii.gz
  │   └── amos_000y.nii.gz
  │   └── ...
  ├── /output/
  │   └── amos_000x.nii.gz
  │   └── amos_000y.nii.gz
  │   └── ...
```
In reality, the cases will not be named with this predictable numbering system. They can have arbitrary file names.

In order to run inference, your training model must be part of a docker image and needs to be added to docker during the build phase of the docker image. Transferring the parameter files is as simple as copying them to a specified folder in the container using the `ADD` command in dockerfile. For more information see the example of docker file we prepared.

Your docker image needs to expose the inference functionality via an inference script which must be named **run_inference.py** and take no additional arguments (must be executable with ``python run_inference.py``). This script needs to use the images provided in ``/input/`` and write the segmentation predictions into the ``/output/`` folder (using the same name as the corresponding input file). 

Under this folder, an example of nnunet framework for docker submission of the AMOS22 challenge are provided.  The nnUNet_submission folder has a dockerfile for running nnUNet baseline model. For final submission,  make sure your inference script should be always called run_inference.py.

### Installation and running guidelines
We recognize that not all participants will have had experience with Docker, so we've prepared quick guidelines for setting up a docker and using the submission examples. Here are the steps to follow to:
- Install docker
- Build a docker image
- Run a container
- Save and load a docker image created

#### Step 1. Install Docker
To install docker use following instructions https://docs.docker.com/engine/install/ depending on your OS.

#### Step 2. Creating Dockerfile
A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. You can find a simple dockerfile in nnUNet_submission/ folder, where we specified requirements needed for running the nnUNet baseline model. Please make sure that your dockerfile is placed in the same folder as your python script to run inference on the test data (run_inference.py) and directory that contains your training weights (parameters/ folder for nnUNet baseline example). Please double check that the naming of your folder with a trained model is correctly specified in a dockerfile as well as in the inference script.

#### Step 3. Build a docker image from a dockerfile
Navigate to the directory with the dockerfile and run following command:

``` bash
docker build -t TEAMNAME .
```
`TEAMNAME` is the unique team name registered by the participant, please use it to name the created docker image, also please include only lowercase letters and no special symbols.

#### Step 4. Run a container from a created docker image
To run a container the docker run command is used:
``` bash
docker run --rm --runtime=nvidia --ipc=host  -e NVIDIA_VISIBLE_DEVICES=all --gpus 0 --user root -v LOCAL_PATH_INPUT:/workspace/input/:ro -v LOCAL_PATH_OUTPUT:/workspace/output/ TEAMNAME python run_inference.py
```
-v flag mounts the directories between your local host and the container. :ro specifies that the folder mounted with -v has read-only permissions. Make sure that LOCAL_PATH_INPUT contains your test samples, and LOCAL_PATH_OUTPUT is an output folder for saving the predictions. During test set submission this command will be run on a private server managed by the organizers with mounting to the folders with final test data. Please test the docker on your local computer using the command above before uploading!

#### Step 5. Save docker image container
To save your docker image to a file on your local machine, you can run the following command in a terminal:
``` bash
docker save TEAMNAME | gzip -c > TEAMNAME.tar.gz
```

This will create a file named TEAMNAME.tar.gz containing your image.

#### Step 6. Load the image
To double check your saved image, you can load it with:
```docker load -i TEAMNAME.tar.gz```

and run the loaded docker as outlined above with the following command (see Step 4):
``` bash
docker run --rm --runtime=nvidia --ipc=host  -e NVIDIA_VISIBLE_DEVICES=all --gpus 0 --user root -v LOCAL_PATH_INPUT:/workspace/input/:ro -v LOCAL_PATH_OUTPUT:/workspace/output/ TEAMNAME python run_inference.py
```

