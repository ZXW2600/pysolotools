import argparse
import json
import logging
import multiprocessing

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

import cv2
import numpy as np
from pycocotools.mask import encode as mask_to_rle



from pysolotools.constants import COCO_KEYPOINTS
from pysolotools.consumers import Solo
from pysolotools.core.models import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotationDefinition,
    Frame,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    KeypointAnnotationDefinition,
    RGBCameraCapture,
    SemanticSegmentationAnnotation,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SOLO2COCOConverter:
    """Convert SOLO to COCO format.
    This converter convert solo format to coco format.
    It supports 4 annotations 2d bbox,keypoints,instance
    and semantic segmentation. Based on annotation types
    of input solo data it does the conversion into coco.


        Examples:
        >>> solo=Solo("src_data_path")
        >>> converter = SOLO2COCOConverter(solo=solo)
        >>> converter.convert(output_path="output/data/path", dataset_name="coco")

    Expected output directory :
    coco:
        └── annotations
            ├── file_name.json
        └── images
    """
    ALL_KEYPOINTS_IN_SOLO:int=-1
    def __init__(self, solo: Solo):
        self._solo = solo
        self._pool = multiprocessing.Pool(processes=4)
        self._bbox_annotations = []
        self._instance_annotations = []
        self._semantic_annotations = []
        self._images = []

    def get_id(self) -> str:
        return "pysolo_extensions.converters.solo2coco"

    @staticmethod
    def _process_rgb_image(image_id, rgb_capture, output, data_root, sequence_num,copy_images,ln_instead_of_cp):
        """
        Args:
            image_id (int): Image ID
            rgb_capture (RGBCameraCapture): RGBCameraCapture object
            output (str): Output directory path
            data_root (str): Data root path
            sequence_num (int): Sequence number

        Returns:
            record: COCO format image record
        """
        width, height = rgb_capture.dimension

        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        image_file = os.path.join(
            data_root, f"sequence.{sequence_num}/{rgb_capture.filename}"
        )

        image_to_file = f"camera_{image_id}.png"
        image_to = image_to_folder / image_to_file
        
        if copy_images:
            if not image_to.exists():
                if ln_instead_of_cp:
                    os.symlink(image_file, str(image_to))
                else:
                    shutil.copy(str(image_file), str(image_to))
            else:
                logger.info(f"Image {image_to_file} already exists.")
            

        record = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_to_file,
        }
        return record

    @staticmethod
    def _load_segmentation_image(filename, data_root, sequence_num):
        """
        Args:
            filename (str): Image file name
            data_root (str): Data root
            sequence_num (int): Sequence number

        Returns:
            img: Segmentation image

        """
        file_path = os.path.join(data_root, f"sequence.{sequence_num}", filename)
        img = cv2.imread(file_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def _compute_segmentation_map(ins_image=None, color=None):
        """
        Args:
            ins_image (): Segmentation image
            color (): RGBA value

        Returns:
            segmentation: Segmentation map
        """

        if ins_image.shape[-1] == 4:
            ins_color = (color[0], color[1], color[2], color[3])
        else:
            ins_color = (color[0], color[1], color[2])

        ins_mask = (ins_image == ins_color).prod(axis=-1).astype(np.uint8)
        segmentation = mask_to_rle(np.asfortranarray(ins_mask))
        segmentation["counts"] = segmentation["counts"].decode()

        return segmentation

    @staticmethod
    def _filter_annotation(ann_type, annotations):
        filtered_ann = list(
            filter(
                lambda k: isinstance(
                    k,
                    ann_type,
                ),
                annotations,
            )
        )
        if filtered_ann:
            return filtered_ann[0]
        return filtered_ann

    @staticmethod
    def _keypoints_map(solo_kp_map: Dict, kp_ann: KeypointAnnotation,keypoints_labels:Optional[Union[List[str],int]]=None):
        """
        Args:
            solo_kp_map (Dict): SOLO keypoint dictionary that maps index to labels
            kp_ann (KeypointAnnotation): KeypointAnnotation object

        Returns:
            kp_map: Dict containing mapping of instance id to COCO format keypoints value and num keypoints tuple.
        """
        keypoint_map = {}
        if keypoints_labels is None:
            keypoints_labels = COCO_KEYPOINTS
        elif keypoints_labels==SOLO2COCOConverter.ALL_KEYPOINTS_IN_SOLO:
            keypoints_labels = [solo_kp_map[i] for i in solo_kp_map.keys()]
        if not isinstance(kp_ann,KeypointAnnotation):
            raise ValueError("kp_ann should be of type KeypointAnnotation")

        for ann_kpt in kp_ann.values:
            keypoints_vals, num_keypoints = [], 0
            for k in keypoints_labels:
                for kpt in ann_kpt.keypoints:
                    label = solo_kp_map[kpt.index]
                    if label == k:
                        keypoints_vals.extend(
                            [
                                float(kpt.location[0]),
                                float(kpt.location[1]),
                                kpt.state,
                            ]
                        )
                        if int(kpt.state) != 0:
                            num_keypoints += 1
                        break

            keypoint_map[ann_kpt.instanceId] = (
                num_keypoints,
                keypoints_vals,
            )
        return keypoint_map

    @staticmethod
    def _sem_class_color_map(sem_seg: SemanticSegmentationAnnotation):
        """
        Args:
            sem_seg (SemanticSegmentationAnnotation): SemanticSegmentationAnnotation object.

        Returns:
            class_color: Dict that maps class to its pixel value.
        """
        class_color = {}
        for ann_seg in sem_seg.instances:
            class_color[ann_seg.labelName] = ann_seg.pixelValue
        return class_color

    @staticmethod
    def _process_annotations(
        image_id, rgb_capture, sequence_num, data_root, solo_kp_map,keypoints_labels
    ):
        bbox_annotations = []
        instance_annotations = []
        semantic_annotations = []

        bbox_ann = SOLO2COCOConverter._filter_annotation(
            ann_type=BoundingBox2DAnnotation, annotations=rgb_capture.annotations
        )
        if bbox_ann:
            bbox_ann =bbox_ann.values

        ins_seg = SOLO2COCOConverter._filter_annotation(
            ann_type=InstanceSegmentationAnnotation, annotations=rgb_capture.annotations
        )

        sem_seg = SOLO2COCOConverter._filter_annotation(
            ann_type=SemanticSegmentationAnnotation, annotations=rgb_capture.annotations
        )

        kp_ann = SOLO2COCOConverter._filter_annotation(
            ann_type=KeypointAnnotation, annotations=rgb_capture.annotations
        )
        if kp_ann:
            keypoint_map = SOLO2COCOConverter._keypoints_map(solo_kp_map, kp_ann,keypoints_labels)
        else:
            keypoint_map = {}

        if sem_seg:
            class_color_map = SOLO2COCOConverter._sem_class_color_map(sem_seg)
            sem_seg_img = SOLO2COCOConverter._load_segmentation_image(
                sem_seg.filename, data_root, sequence_num
            )

            for bbox in bbox_ann:
                x, y, w, h = [
                    float(bbox.origin[0]),
                    float(bbox.origin[1]),
                    float(bbox.dimension[0]),
                    float(bbox.dimension[1]),
                ]
                segmentation = []
                if bbox.labelName in class_color_map:
                    segmentation = SOLO2COCOConverter._compute_segmentation_map(
                        ins_image=sem_seg_img.copy(),
                        color=class_color_map[bbox.labelName],
                    )
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": segmentation,
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                }
                semantic_annotations.append(record)

        if ins_seg:

            ins_seg_img = SOLO2COCOConverter._load_segmentation_image(
                ins_seg.filename, data_root, sequence_num
            )
            ins_seg_instances = ins_seg.instances

            if len(bbox_ann) != len(ins_seg_instances):
                raise ValueError("All bounding boxes does not have segmentation map.")

            for bbox, ins in zip(bbox_ann, ins_seg_instances):

                x, y, w, h = [
                    float(bbox.origin[0]),
                    float(bbox.origin[1]),
                    float(bbox.dimension[0]),
                    float(bbox.dimension[1]),
                ]
                segmentation = SOLO2COCOConverter._compute_segmentation_map(
                    ins_image=ins_seg_img.copy(), color=ins.color
                )
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": segmentation,
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                }
                instance_annotations.append(record)

        else:

            for bbox in bbox_ann:
                x, y, w, h = [
                    float(bbox.origin[0]),
                    float(bbox.origin[1]),
                    float(bbox.dimension[0]),
                    float(bbox.dimension[1]),
                ]
                keypoints_vals, num_keypoints = [], 0
                if kp_ann and bbox.instanceId in keypoint_map:
                    num_keypoints, keypoints_vals = keypoint_map[bbox.instanceId]

                record = {
                    "segmentation": [],
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "keypoints": keypoints_vals,
                    "num_keypoints": num_keypoints,
                    "category_id": bbox.labelId,
                    "id": bbox.instanceId,
                }
                bbox_annotations.append(record)

        return bbox_annotations, instance_annotations, semantic_annotations

    def _categories(self):
        categories = []
        ann_defs = self._solo.annotation_definitions.annotationDefinitions
        bbox_ann_def = list(
            filter(
                lambda ann_def: isinstance(ann_def, BoundingBox2DAnnotationDefinition),
                ann_defs,
            )
        )[0]
        for label_spec in bbox_ann_def.spec:
            record = {
                "id": label_spec.label_id,
                "name": label_spec.label_name,
                "supercategory": "default",
                "keypoints": [],
                "skeleton": [],
            }
            categories.append(record)

        return categories

    @staticmethod
    def _create_ann_file(data, output_dir, file_name):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_dir}/{file_name}"
        with open(output_file, "w") as out:
            json.dump(data, out, indent=3)

    def _write_out_annotations(self, output_dir):
        date_time = datetime.now()
        date_created = date_time.strftime("%A, %d %B, %Y")
        instances = {
            "info": {
                "year": datetime.now().year,
                "version": "0.0.1",
                "description": "COCO compatible Synthetic Dataset",
                "contributor": "Anonymous",
                "url": "Not Set",
                "date_created": date_created,
            },
            "licences": [{"id": 0, "name": "No License", "url": "Not Set"}],
            "images": self._images,
            "annotations": [],
            "categories": self._categories(),
        }
        if self._semantic_annotations:
            for idx, ann in enumerate(self._semantic_annotations):
                ann["id"] = idx
            instances["annotations"] = self._semantic_annotations
            self._create_ann_file(instances, output_dir, "semantic.json")

        if self._instance_annotations:
            for idx, ann in enumerate(self._instance_annotations):
                ann["id"] = idx
            instances["annotations"] = self._instance_annotations
            self._create_ann_file(instances, output_dir, "instances.json")
        else:
            for idx, ann in enumerate(self._bbox_annotations):
                ann["id"] = idx
            instances["annotations"] = self._bbox_annotations
            self._create_ann_file(instances, output_dir, "bbox.json")

    @staticmethod
    def _process_instances(
        frame: Frame, idx, output, data_root, solo_kp_map,copy_images,ln_instead_of_cp,keypoints_labels
    ) -> Tuple[Dict, List, List, List]:
        logger.info(f"Processing Frame number: {idx}")
        image_id = idx
        sequence_num = frame.sequence
        rgb_capture = list(
            filter(lambda cap: isinstance(cap, RGBCameraCapture), frame.captures)
        )[0]

        img_record = SOLO2COCOConverter._process_rgb_image(
            image_id, rgb_capture, output, data_root, sequence_num,copy_images,ln_instead_of_cp
        )
        (
            ann_record,
            ins_ann_record,
            sem_ann_record,
        ) = SOLO2COCOConverter._process_annotations(
            image_id, rgb_capture, sequence_num, data_root, solo_kp_map,keypoints_labels
        )

        return img_record, ann_record, ins_ann_record, sem_ann_record

    def _get_solo_kp_map(self):
        solo_kp_map = {}
        kp_ann_def = list(
            filter(
                lambda k: isinstance(
                    k,
                    KeypointAnnotationDefinition,
                ),
                self._solo.annotation_definitions.annotationDefinitions,
            )
        )
        if kp_ann_def:
            for kp in kp_ann_def[0].template.keypoints:
                solo_kp_map[kp.index] = kp.label
        return solo_kp_map

    def callback(self, result):
        self._images.append(result[0])
        self._bbox_annotations += result[1]
        self._instance_annotations += result[2]
        self._semantic_annotations += result[3]
    def error_callback(this,e):
        print(f"Error: {e}")
        raise Exception(e)

    def convert(self, output_path: str, dataset_name: str = "coco",copy_images:bool=True,ln_instead_of_cp:bool=False,keypoints_labels:Optional[Union[List[str],int]]=None):
        """_summary_

        Args:
            output_path (str): _description_
            dataset_name (str, optional): _description_. Defaults to "coco".
            copy_images (bool, optional): _description_. Defaults to True.
            ln_instead_of_cp (bool, optional): _description_. Defaults to False.
            keypoints_labels (Optional[List[str]], optional): which keypoints to include, if empty, using COCO Keypoints labels. Defaults to None.
        """        
        output = os.path.join(output_path, dataset_name)

        solo_kp_map = self._get_solo_kp_map()

        for idx, frame in tqdm(enumerate(self._solo.frames()),total=self._solo.end or self._solo.metadata.totalFrames):
            self._pool.apply_async(
                self._process_instances,
                args=(frame, idx, output, self._solo.data_path, solo_kp_map,copy_images,ln_instead_of_cp,keypoints_labels),
                callback=self.callback,
                error_callback=self.error_callback
            )
            
        self._pool.close()
        self._pool.join()

        self._write_out_annotations(output)


def cli():
    parser = argparse.ArgumentParser(
        prog="solo2coco",
        description=("Converts SOLO datasets into COCO datasets",),
        epilog="\n",
    )

    parser.add_argument("solo_path")
    parser.add_argument("coco_path")
    parser.add_argument(
        "-n", "--name", default="coco", help="The name of the coco dataset"
    )

    args = parser.parse_args(sys.argv[1:])

    solo = Solo(args.solo_path)

    converter = SOLO2COCOConverter(solo)

    converter.convert(output_path=args.coco_path, dataset_name=args.name)


if __name__ == "__main__":
    cli()
