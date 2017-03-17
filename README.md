# Food-Image-Retrieval-System
Retrieve images of your favourite food!

Refer to [the report](https://github.com/HiiYL/Food-Image-Retrieval-System/blob/master/Report-Assignment2.pdf) for more information.

## Installation
##### Note: Installation has been tested on python 2.7 and python 3.4, if you encounter any issues, please first test if it works with the two versions.
1. (Optional) Fork the repository.
2. Clone the repository.
3. ` conda install -c menpo cyvlfeat opencv3 h5py theano`
4. ` pip install -r requirements.txt `
5. [Windows only] ` conda install -c msys2 m2w64-gcc=5.3.0 `

## Usage
Once all requirements have been satisfied, run scripts in following order.

1. ` python featureExtraction.py `
2. ` python featureExtractionDeep.py `
3. ` python fullEval.py `



## Performance
![figure_1](https://cloud.githubusercontent.com/assets/7908951/24045271/72bb0836-0b59-11e7-8c84-c2e5c3ca3e84.png)

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D
