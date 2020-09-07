# Bonsai Connectors

![Alt Text](assets/cart_pole.gif)|![Alt Text](assets/mountain_car.gif)|![Alt Text](assets/hoppers.gif)  |![Alt Text](assets/reacher.gif)
:-------------------------------:|:----------------------------------:|:-------------------------------:|:------------------------------:


Bonsai Connectors is an open-source library, which gives us access to OpenAI Gym standardised set of environments while using Microsoft's reinforcement learning platform Bonsai.

The repository also contains examples how to use this library to build and deploy OpenAI Gym environments to Bonsai and how to interact with the trained agent (Brain) from your code.  

## Basics

There are two basic concepts in reinforcement learning: the environment (namely, the outside world) and the agent (the algorithm you are writing). The agent sends actions to the environment, and the environment replies with observations and rewards (that is, a score).

OpenAI Gym is a toolkit for developing simulations and comparing reinforcement learning algorithms. The Gym open-source library gives us access to a standardised set of environments. Environments come as is with no predefined agent.

Link to Open AI environments: https://github.com/openai

Bonsai is the machine teaching service in the Autonomous Systems suite from Microsoft. It builds on innovations in reinforcement learning to simplify AI development.
We use Bonsai to create agents (brains) that control and optimise complex systems. No neural net design required.

Full documentation for Bonsai's Platform can be found at https://docs.bons.ai.

## Set-Up

We are using Python 3.8.3, you might need to use python3 command if you are running multiple versions.

You will need to create an account with Microsoft Bonsai.
Follow instructions: https://docs.microsoft.com/en-us/bonsai/guides/account-setup

Bonsai Connectors requires two environment variables to be set to be able to connect to Microsoft Bonsai:

**SIM_ACCESS_KEY**. You can copy it from the Account Settings page.

**SIM_WORKSPACE**. You can find this in the URL after ***/workspaces/*** once you are logged in to the platform.

You will need to install support libraries prior to running locally.
Our environment depend on **microsoft_bonsai_api** package and on **gym_connectors** from this codebase.

```
cd connectors
pip install .
pip install microsoft_bonsai_api
```

For the PyBullet environments you will need additionally the **pybullet-gym**. 
We have added a default arena and fixed an issue with the camera, so we advise to use are forked version. Original code can be found here: https://github.com/benelot/pybullet-gym
The flag -e in pip is required to install the assets.

```
git clone https://github.com/myned-ai/pybullet-gym.git
cd pybulley-gym]
pip install -e .
```


### Building Dockerfile
To upload and use the simulator from Azure, you need to push it as a docker image to Azure Container Registry.

Clone the repo and go into the created folder and select an environment, e.g:

```
cd ./envs/classic_controls/Pendulum
```

From the root of the selected environment run:

```
docker build -t <IMAGE_NAME> -f Dockerfile ../../../
```

### Push to ACR
Run the following code to push to ACR:

```
az login
az acr login --subscription <SUBSCRIPTION_ID> --name <ACR_REGISTRY_NAME>
docker tag <IMAGE_NAME> <ACR_REGISTRY_NAME>.azurecr.io/bonsai/<IMAGE_NAME>
docker push <ACR_REGSITRY_NAME>.azurecr.io/bonsai/<IMAGE_NAME>
```

### Create Simulator in Bonsai
Once you have pushed your docker image to ACR, you can create a simulator by clicking ***Add Sim*** from the left hand side navigation menu. Enter the ACR URL of the image and name it.

### Create Brain in Bonsai
You can create a brain by clicking ***Create brain*** from the left hand side navigation menu. Select ***Empty Brain***, add a name and after it has been created, copy the contents of the .ink file from the selected environment (folder) and paste them to the ***Teach*** section of the brain. Click the train button and from the presented list select the simulator you have created in the previous step.

### Running Local Agent
When you are satisfied with the training progress, stop the training and export the brain.
Run the presented code to download the exported docker image locally.

Start the agent.py located on the root of your selected environment.
The Open AI visualiser of your selected environment will start and you will see how well your trained brain 'behaves'.

## Environments

We have developed few working examples and we aim to expand this list continuously by adding new environments from different physics engines.
As with every problem, there are more than just one way to solve or achieve satisfactory results.
We are open to suggestions and we encourage code contribution.

Inside on each environment folder we have created an agent that can run locally to communicate with the exported Bonsai brain (running on Docker) and is rendering the simulation using Open AI environment.

### Classic Controls

A collection of control theory problems from the classic RL literature.

[README link](https://github.com/myned-ai/bonsai-connectors/blob/tide-up/envs/classic_controls/README.md)

### PyBullet

Bullet is a physics engine which simulates collision detection, soft and rigid body dynamics.

PyBullet Gymperium is an open-source implementation of the OpenAI Gym MuJoCo environments for use with the OpenAI Gym Reinforcement Learning Research Platform in support of open research.

[README link](https://github.com/myned-ai/bonsai-connectors/blob/tide-up/envs/pybullet/README.md)