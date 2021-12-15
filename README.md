# BarcodeDetection
This repository is a GPGPU project directed by EPITA courses with pedagogical purposes. The objective is to develop a codebar detector based on CUDA parrallelized framework.

## MEMBERS:

* Kevin GUILLET
* Corentin BUNEL
* Maxime LEHERLE
* Antoine Julien
* Victor Dutto

## HOW TO USE:

usage of the detector binary:

./detector `<path/to/image.jpg>` [OPTIONS]


informations:
* Options are described by the command:
./detector --help
* run CPU baseline with step image by:
./detector image.jpg --cpu --step
* run GPU baseline with step image save in out/:
./detector image.jpg --gpu --step --blocks 6000 --threads 2048

## INSTALLATION:

run the following commands:
```sh
mkdir build
cd build
cmake ..
make -j4
```
In case of error related to google benchmark file, open the concerned file and add this line of include:

```c
#include <limits>
```

and re-run:
```sh
make -j4
```

And enjoy the code.
