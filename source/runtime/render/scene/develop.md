## TODO

1. Scene Texture memagement
2. Loaded resource CPU representation，except Index/Vertex Buffer & Texture Images (GPU).
    * Geometry Info，Instance Info，Geometry - Instance
    * Light Info
    * Material Info，Mesh - Material Info
      * 一个Mesh对应多个Geometry，Geometry对应材质，应该考虑按Geometry level上传index/vertex buffer，进行去重
    * etc.
    * MeshComponent
      * 一个mesh component对应一个transform component，即instance info
      * 包含多个Geometry，多个Geometry Info，一个geometry info对应一个material info
      * geometry info：
        * index/vertex buffer handle
        * material handle
      * mesh info
3. Upload resources to GPU.