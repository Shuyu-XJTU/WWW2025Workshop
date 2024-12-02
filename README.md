# WWW2025 Workshop

**The PAB dataset is released at [Baidu Yun](https://pan.baidu.com/s/1gqY6DuTL-EStXlH0dz05ng) [mdjb] and [OneDrive](https://1drv.ms/f/c/afc02d7952f9b34d/Epb3qCEwsMJOjYIx-sMm_rkBbZfyiD8I-bRmLp0X-rT1vQ?e=7gyGco).**


**Some details of PAB:**

* "image": image dir

* "caption": generated image caption by Qwen2-VL

* "image_id": "i_j", <-- pair_i.json & image-text index in this json: j

* "hard_i": hard negative image for caption

* "hard_c": hard negative text for image

* "hard_i_id": the id of the matched image-text pair (hard_i, hard_c) 

* "source_id": "x_y", <-- x: OOPS video id; if y=0, source_caption is Cn, elif y=1, source_caption is Ca, elif y=2, source_caption is Ca+

* "source_caption": caption from corresponding OOPS video (image is generated by this source caption)


Attribute annotation and baseline code is coming soon.
