PARA ABRIR EL DOCKER:
--  docker compose -f docker/docker-compose.yml run --rm tfg-sim /bin/bash

mario@mario-AERO-17-XE5:~/tfg/vlmap-semantic-object-search-tfg$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.5 LTS
Release:	22.04
Codename:	jammy


mario@mario-AERO-17-XE5:~/tfg/vlmap-semantic-object-search-tfg$ nvidia-smi
Wed Mar 11 12:58:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   49C    P0            751W /   80W |      15MiB /   8192MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1886      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+

mario@mario-AERO-17-XE5:~$ source ~/.bashrc
(base) mario@mario-AERO-17-XE5:~$ conda --version
conda 26.1.1

mario@mario-AERO-17-XE5:~$ docker --version
docker compose version
Docker version 29.3.0, build 5927d80
Docker Compose version v5.1.0


