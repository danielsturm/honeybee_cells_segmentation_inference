# Annotation Tool

- Uses Napari as framework (link)
- Has three main components the cell finder and the annotation tool and the segmentation mask creator

## How to use

### Cell finder
1. run the Cell Finder to find as many cells as possible beforehand.
    - This will created a json file with the found cells, with the same name as the image

### Annotation Tool
    - Each image needs a json file with pre detected cells
    - If an image does not have a corresponding labels json, an empty one will be created
    - run the tool from the terminal with

    - if you like you can enable tooltips for the cells layer that display the label and diameter
        - to to File -> Preferences -> Appearance -> Show layer tooltips

    - select points by using the brush in the brush layer or the mouse in the points layer
    - use shortcuts to activate the points/cell layer (hit 'c') or the brush layer (hit 'b')
    - change the point size by hitting Alt and scroll up or down

### Segmentation Mask Creator
