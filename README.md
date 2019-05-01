# face-tracking
Face tracking program using registration.

## bag2imgs folder
bag2imgs.py can convert .bag to RGB and depth images.

## face tracking folder
facetrack.py can tracking face area.

when facetrack.py is run, you need face++(https://www.faceplusplus.com/) account.
cpp relations is under.

・align.cpp -> alignDepth.exe

・deproj.cpp -> projection.exe

・depth2ply.cpp -> depth2Img2ply.exe

# Dependencies
## bag2imgs
・rosbag

・opencv-python v4.0




## face tracking programs
### python
・open3d-python

・opencv-python v4.0

### c++
・opencv v4.0
