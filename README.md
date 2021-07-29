# End Credits Detection

Machine Learning model to detect the start time of end credits on series and movies

### Instalation

The project requires Python to be installed. 

The detector is available both directly through source code and as a python module, which can be used from console and inside other appllications. 
To install the module:
```pip install credits-detector/dist/credictor-0.3.0.tar.gz```

### Usage

To detect the start time in milliseconds of the end credits of a given video:

```
from credictor import CreditsPredictor

detector = CreditsPredictor()
time = detector.predict('path_to_video')
```
