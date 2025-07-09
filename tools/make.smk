bboxes = config["bboxes"]
keypoint = config["keypoint"]
classify = config["classify"]
add = config["add"]


yolo_model = "output/bee_detection_final.pt"

rule extract_bboxes:
    input:
        model=yolo_model,
        images="{image_path}",
        labels="{label_dir}"
    output:
        csv="{csv_file}"
    script:
        bboxes
rule keypoint_detection:
    input:
        csv="{csv_file}"
    output:
        csv="{csv_file}" 
    script:
        detection
rule classify_species:
    input:
        csv="{csv_file}"
    output:
        csv="{csv_file}"  
    script:
        classify
rule add_model_weights:
    input:
        csv="{csv_file}"
    output:
        csv="{csv_file}" 
    script:
        add

rule all:
    input:
        "{csv_file}"
