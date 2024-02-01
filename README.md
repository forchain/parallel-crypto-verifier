# Parallel Blockchain Transaction Verifier

## Intro
This master graduation project focuses on employing parallel computing acceleration to enhance the efficiency of the Elliptic Curve Digital Signature Algorithm (ECDSA) using the secp256k1 curve. Elliptic curve cryptography is a key component in blockchain technologies, favored for its robust security and efficiency in resource utilization. Our initiative significantly boosts the processing speed of signature verifications for this algorithm by leveraging the power of multicore acceleration.

The implementation utilizes the OpenCL framework, a cross-platform computing interface that allows compatibility across a diverse range of hardware. This approach ensures that our solution can be deployed on GPUs from various manufacturers, enhancing its versatility and practical applicability. In addition, it utilizes OpenMPI framework for the cluster computing setup.


### Notes
##### OpenMP And MPI
* Ubuntu:
```
sudo apt install mpich
```
* MacOS:
```
brew install llvm
brew install libomp
brew install mpich
```

##### Environmental deployment for MPI
* Ensure that MPI has been installed
```
sudo apt install mpich
mpichversion
```

* Ensure that the same directory for each machine has applications and data. Recommend using NFS service.
```
server:
sudo apt install nfs-kernel-server

sudo mkdir /ECDSA-OpenCL
sudo chmod -R 777 /ECDSA-OpenCL
sudo chown root:root /ECDSA-OpenCL/ -R

~ pwd
/ECDSA-OpenCL
~ ls
verifier
hosts
data/signature_data.csv

sudo /etc/init.d/nfs-kernel-server start

client:
sudo apt install nfs-common
sudo mount -t nfs ip.ip.ip.ip:/ECDSA-OpenCL /mnt -o nolock

```
* Ensure that each machine can support password free login by ssh.
```
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub  root@x.x.x.x   # Install public keys on all client machines

sudo vim /etc/ssh/sshd_config

PubkeyAuthentication yes

sudo systemctl restart sshd

```
* Running application
```
./verifier -a
or
mpirun -np 5 ./verifier --mpicpu
or
mpirun -hostfile hosts -np 5 ./verifier --mpicpu
```

* Running application for dynamically assigning tasks

```
mpirun -np 3 ./verifier -D
or
mpirun -hostfile hosts -np 5 ./verifier -D
```

## Compilation

``` sh
chmod u+x *
./autogen.sh 
./configure
make
./verifier -h
```
