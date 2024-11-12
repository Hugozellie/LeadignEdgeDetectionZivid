import zivid    as z
import numpy    as np
import datetime as dt
import open3d   as o3d
import matplotlib.pyplot as plt
from collections import Counter



def connectCamera():
    # try and connect to the next available camera
    try:
        print("connecting")
        app = z.Application()
        cameras = app.cameras()
        for c in cameras:
            print(c.state)
        camera = app.connect_camera()
        print("connected")

        return camera
    # if no camera is available, exit the program
    except:
        print("unable to connect camera.\nCheck if camera is not busy.")
        exit()

def capture(camera: z.Camera):
    # set the settings as seen bellow
    print("setting settings")
    settings = z.Settings()
    settings.acquisitions.append(z.Settings.Acquisition())
    settings.acquisitions[0].aperture = 4.0
    settings.acquisitions[0].exposure_time = dt.timedelta(microseconds=30000)
    settings.acquisitions[0].gain = 1.5
    settings.acquisitions[0].brightness = 1.5

    settings.processing.Filters.Cluster.Removal.enabled = True
    settings.processing.Filters.Cluster.Removal.max_neighbor_distance = 5.0
    settings.processing.Filters.Cluster.Removal.min_area = 200.0

    settings.processing.Filters.Hole.Repair.enabled = True
    settings.processing.Filters.Hole.Repair.hole_size = 0.1
    settings.processing.Filters.Hole.Repair.strictness = 2

    settings.processing.Filters.Noise.Removal.enabled = True
    settings.processing.Filters.Noise.Removal.threshold = 5.0

    settings.processing.Filters.Outlier.Removal.enabled = True
    settings.processing.Filters.Outlier.Removal.threshold = 7.0

    settings.processing.Filters.Reflection.Removal.enabled = True
    settings.processing.Filters.Reflection.Removal.mode = "Local"

    settings.processing.Filters.Smoothing.Gaussian.enabled = True
    settings.processing.Filters.Smoothing.Gaussian.sigma = 1.0
    
    settings.processing.Filters.Experimental.ContrastDistortion.Correction.enabled = True
    settings.processing.Filters.Experimental.ContrastDistortion.Correction.strength = 0.3
    print("settings set")

    # capture the current scan using the above settings and save it
    with camera.capture(settings) as frame:
        frame.save("result.zdf")
        print("captured")

def loadScan(path: str):
    # load the .zdf file at the given path
    print(f"Reading point cloud from file: {path}")
    frame = z.Frame(path)
    point_cloud = frame.point_cloud()

    # copy the xyz and rgba data from the .zdf file
    xyzLoad = point_cloud.copy_data("xyz")
    rgbaLoad = point_cloud.copy_data("rgba")

    # reshape to a 2d numpy array
    rgbaLoad = rgbaLoad.reshape(-1,rgbaLoad.shape[-1])
    xyzLoad = xyzLoad.reshape(-1, xyzLoad.shape[-1])

    # remove invalid points
    rgbaLoad = rgbaLoad[~np.isnan(xyzLoad).any(axis=1)]
    xyzLoad = xyzLoad[~np.isnan(xyzLoad).any(axis=1)]

    # remove the 'a' row from the rgba data
    rgbLoad =  np.delete(rgbaLoad, (3), axis=1)

    return xyzLoad, rgbLoad

def exportData(dataxyz,datargb):
    # saves the xyz and rgb data to .npz files to access them elsewhere. Mainly used to export data for the importData() function.
    np.savez("Zivid3dSampleData.npz", xyz = dataxyz, rgb = datargb)

def importData(path: str):
    # imports xyz and rgb data from a .npz file. Mainly used to keep working with the same data used by the exportData() function.
    try:
        loaded = np.load(path)
    except:
        print("File path incorrect. Exiting.")
        exit()
    
    # try and load the xyz data. If it fails it exits the program
    try:
        loadedxyz = loaded["xyz"]
    except:
        print("Failed to load xyz data from the given file. Exiting.")
        exit()
    
    # try and load the rgb data. If it fails it exits the program
    try:
        loadedrgb = loaded["rgb"]
    except:
        print("Failed to load rgb data from the given file. Exiting.")
        exit()

    return loadedxyz, loadedrgb

def makePointCloud(xyz: np.ndarray, rgb: np.ndarray = [0]):
    # tries to add the xyz data to the point cloud, otherwise exits the program
    try:
        xyzcopy = np.vstack([xyz, [0,0,0]])
        pcdCopy = o3d.geometry.PointCloud()
        pcdCopy.points = o3d.utility.Vector3dVector(xyzcopy)
    except:
        print("xyz data for point cloud is wrong.")
        exit()

    # checks if the given rgb data matches the length of the xyz data, otherwise exits the program
    if(len(rgb) == len(xyz)):
        try:
            rgbcopy = np.vstack([rgb, [0,0,0]])
            pcdCopy.colors = o3d.utility.Vector3dVector(rgbcopy.astype(float) / 255.0)
        except:
            print("color data for point cloud is wrong.")
            exit()

    return pcdCopy

def viewPointCloud(*pcdref: o3d.cpu.pybind.geometry.PointCloud):
    viewer = o3d.visualization.Visualizer()

    for i in range(len(pcdref)):
    # copy the reference point cloud as to not make changes to the original
        pcdCopy = o3d.geometry.PointCloud(pcdref[i])

        # add a rotation to the pointcloud so that it matches the view window
        R = pcdCopy.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
        pcdCopy.rotate(R, center=(0, 0, 0))

        # creates the viewer and displays the point cloud
        viewer.create_window()
        viewer.add_geometry(pcdCopy)

    opt = viewer.get_render_option()
    opt.show_coordinate_frame = False # change to True if the global axis needs to be shown
    viewer.run()
    viewer.destroy_window()

def clusterPointCloud(pcdref: o3d.cpu.pybind.geometry.PointCloud, eps: float, minPoints: int, log: bool = False):
    # copy the reference point cloud as to not make changes to the original
    pcdCopy = o3d.geometry.PointCloud(pcdref)
    
    # Apply DBSCAN clustering
    labels = np.array(pcdCopy.cluster_dbscan(eps=eps, min_points=minPoints, print_progress=log))
    maxLabel = labels.max()

    # Colorize clusters
    colors = plt.get_cmap("tab20")(labels / (maxLabel if maxLabel > 0 else 1))
    colors[labels < 0] = 0  # Noise points are assigned a color of black
    pcdCopy.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcdCopy

def largestClusterPointCloud(pcdref: o3d.cpu.pybind.geometry.PointCloud, eps: float, minPoints: int, log: bool = False):
    # copy the reference point cloud as to not make changes to the original
    pcdCopy = o3d.geometry.PointCloud(pcdref)

    # Apply DBSCAN clustering
    labels = np.array(pcdCopy.cluster_dbscan(eps=eps, min_points=minPoints, print_progress=log))

    clusterCount = Counter(labels)
    largestClusterLabel = clusterCount.most_common(1)[0][0]  # Get the label of the largest cluster
    
    # Extract points that belong to the largest cluster
    largestClusterIndices = np.where(labels == largestClusterLabel)[0]
    pcdCopy = pcdCopy.select_by_index(largestClusterIndices)

    return pcdCopy

def findPointCloudEdge(pcdref: o3d.cpu.pybind.geometry.PointCloud):
    # copy the reference point cloud as to not make changes to the original
    pcdCopy = o3d.geometry.PointCloud(pcdref) 

    # Estimate normals
    pcdCopy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=50))

    # Set up KDTree for neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcdCopy)

    # Compute curvature (approximated by normal changes)
    normals = np.asarray(pcdCopy.normals)
    curvature = np.zeros(normals.shape[0])

    # Compute curvature by measuring normal angle difference with neighbors
    for i in range(len(normals)):
        # Search for nearest neighbors for the current point
        [k, idx, _] = kdtree.search_knn_vector_3d(pcdCopy.points[i], 50)
        # Calculate the average normal of neighbors
        avg_normal = np.mean(normals[idx], axis=0)
        # Curvature as the difference between the point's normal and the average neighbor's normal
        curvature[i] = np.linalg.norm(normals[i] - avg_normal)

    # Mark the edges
    threshold = np.percentile(curvature, 98) 
    edge_indices = np.where(curvature > threshold)[0]

    # Create a new point cloud for edges
    edges = pcdCopy.select_by_index(edge_indices)

    # Color the edge points
    pcdCopy.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(pcdCopy.points))])  # Blue for non-edges
    edges.paint_uniform_color([1, 0, 0])  # Red for edges

    return pcdCopy, edges


#TO RUN
cam = connectCamera()
capture(cam)

xyz, rgb = loadScan("result.zdf")
exportData(xyz,rgb)

xyz, rgb = importData("Zivid3dSampleData.npz")

pcd = makePointCloud(xyz,rgb)
pcdCluster = clusterPointCloud(pcd, eps=1.32, minPoints=10, log=True)
pcdLargest = largestClusterPointCloud(pcd, eps=1.32, minPoints=10, log=True)
_,pcdEdges = findPointCloudEdge(pcdLargest)

pcdEdgesCluster = clusterPointCloud(pcdEdges, eps=2.0, minPoints=10, log=True)
pcdLargestEdge = largestClusterPointCloud(pcdEdgesCluster, eps=2.0, minPoints=10, log=True)

viewPointCloud(pcdLargest, pcdLargestEdge)
