# Example use of pointersect_full

We demonstrate how to use `pointersect_full` to render point clouds 
sampled from a mesh, which allows us to compute ground truth.

```bash
# in the folder
# download tex-model meshes
bash download.sh

# render the mesh
pointersect_full --config_filename config_mesh.yaml
```

##### Mesh Credit
```
@article{Texturemontage05,
    author = "Kun Zhou and Xi Wang and Yiying Tong and Mathieu Desbrun and Baining Guo and Heung-Yeung Shum",
    title = "Texturemontage: Seamless Texturing of Arbitrary Surfaces From Multiple Images",
    journal = "ACM Transactions on Graphics",
    volume = "24",
    number = "3",
    year="2005",
    pages = "1148-1155"
}
```