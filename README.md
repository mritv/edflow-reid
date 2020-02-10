# edflow-reid
ReID callback for evaluation of edflow projects. 
ReID implementation from https://github.com/VisualComputingInstitute/triplet-reid


## edflow Evaluation
In order to use this callback in your edflow project, install this repository by cloning it and executing 
`pip install edflow-reid`.

You can then call this callback on your edflow evaluation data with the help of the following command:

```commandline
edflow -n <name-of-your-evaluation> -b <path-to-your-config> -p <path-to-the-project> -e 
```

with this in your config:

```yaml
eval_pipeline:
    callbacks:
        reid: edflow-reid.reid_callback.reid_callback
      
    callback_kwargs:
        reid:
            im_in_key: "image_in"
            im_out_key: "image_out"
            name: "reid"
```

Or, you can call it on already created data using `edeval`

```commandline
edeval -c <path-to-model_outputs.csv> -cb reid: edflow-reid.reid_callback.reid_callback
```