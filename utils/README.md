**************************************************************************************************************
# UTILITIES
**************************************************************************************************************

## ram_usage.py

This script will report to you how long of a sequence you can embed at once given your system's memory constraints. This is useful for determining the `--maxlen` parameter for the make_db.py script. You can run the script with the following command from the parent directory:

```
python -m utils.ram_usage
```

This will test subsequences of different lengths and report what is the maximum length that can be embedded at once. It will by default test lengths up to 2000, but if you have a particularly large system, you can specify a larger length to test with the `--testmax` parameter.