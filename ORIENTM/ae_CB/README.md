# AE data(Boulder)

AE data from CU Boulder group, the website is [here](http://lasp.colorado.edu/space_weather/dsttemerin/dsttemerin.html)
This data stop updating since Jan 8th, 2021

## Load data
```python
import SWO2R
from datetime import datetime, timedelta

start_time =datetime.utcnow() - timedelta(days = 60)
end_time = datetime.utcnow() - timedelta(days = 30)
SWO2R.ae_CB.load(start_time, end_time)
```
