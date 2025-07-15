# Use ```.vrs``` files

## Extract images from ```.vrs``` files

1. Install the required vrs package (see [vrs documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/aria_vrs_tools_installation)):
```bash
conda install vrs --channel=conda-forge
```

2. Run this command to extract images:
```bash
vrs extract-images <vrs-file-name> --to <output-directory>
```