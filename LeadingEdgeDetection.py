import zivid    as z
import numpy    as np
import datetime as dt
import open3d   as o3d
from tqdm import tqdm
from collections import Counter
from rdp import rdp
from scipy.spatial import distance_matrix, KDTree


class setup:
    def __init__(self):
        self.app = z.Application()
        self.settings = z.Settings()

    def connectCamera(self, log:bool = False):
        """ Tries and connect to the first available camera. """
        try:
            cameras = self.app.cameras()

            if(log == True): 
                # show a list of all available cameras
                print("connecting")
                for c in cameras:
                    print(c.state)
                
            # connect to the camera
            self.camera = self.app.connect_camera()

            if(log == True):
                print("connected")
        except:
            print("unable to connect camera.\nCheck if camera is not busy.")
            exit()
    
    def setSettings(self, log:bool = False):
        """ sets the settings for the connected camera. """
        if(log == True):
            print("configuring settings")

        # checks if settings have already been set, otherwise set the settings 
        if(len(self.settings.acquisitions) == 0):
            self.settings.acquisitions.append(z.Settings.Acquisition())
            self.settings.acquisitions[0].aperture = 4.0
            self.settings.acquisitions[0].exposure_time = dt.timedelta(microseconds=75000)
            self.settings.acquisitions[0].gain = 1.0
            self.settings.acquisitions[0].brightness = 1.0

            self.settings.processing.Filters.Cluster.Removal.enabled = True
            self.settings.processing.Filters.Cluster.Removal.max_neighbor_distance = 7.0
            self.settings.processing.Filters.Cluster.Removal.min_area = 500.0

            self.settings.processing.Filters.Hole.Repair.enabled = True
            self.settings.processing.Filters.Hole.Repair.hole_size = 0.2
            self.settings.processing.Filters.Hole.Repair.strictness = 2

            self.settings.processing.Filters.Noise.Removal.enabled = True
            self.settings.processing.Filters.Noise.Removal.threshold = 5.0
            self.settings.processing.Filters.Noise.Suppression = True
            self.settings.processing.Filters.Noise.Repair = True

            self.settings.processing.Filters.Outlier.Removal.enabled = True
            self.settings.processing.Filters.Outlier.Removal.threshold = 7.0

            self.settings.processing.Filters.Reflection.Removal.enabled = True
            self.settings.processing.Filters.Reflection.Removal.mode = "Local"

            self.settings.processing.Filters.Smoothing.Gaussian.enabled = True
            self.settings.processing.Filters.Smoothing.Gaussian.sigma = 1.0
                
            self.settings.processing.Filters.Experimental.ContrastDistortion.Correction.enabled = True
            self.settings.processing.Filters.Experimental.ContrastDistortion.Correction.strength = 0.3

            self.settings.processing.Color.Balance.red   = 1.0
            self.settings.processing.Color.Balance.green = 1.0
            self.settings.processing.Color.Balance.blue  = 1.8 

        if(log == True):
            print("settings configured")    

    def disconnectCamera(self):
        """ Disconnects the connected camera. """
        self.camera.disconnect()

    def getSettings(self):
        """ Returns the settings of the connected camera. """
        return self.settings
    
    def getApplication(self):
        """ Returns the application instance of the connected camera. """
        return self.app
    
    def getCamera(self):
        """ Returns the camera instance of the connected camera. """
        return self.camera

class scan:
    """ Class to scan and export data from the zivid camera. """
    def __init__(self, amount:int = 1):
        self.amount = amount

    def capture(self, camera, settings, name:str = None, ret:bool = False, log:bool = False):
        """ Captures scans using the given camera and the given settings. The number of scans made is defined upon initialization. """
        i = 0

        if(log == True):
            print("capturing")

        # scan until the batch amount has been reached
        while(i < self.amount):
            with camera.capture(settings) as frame:
                pcdData = frame.point_cloud()

                # copy the xyz and rgba data from the .zdf file
                xyzLoad = pcdData.copy_data("xyz")
                rgbaLoad = pcdData.copy_data("rgba")

                # reshape to a 2d numpy array
                rgbaLoad = rgbaLoad.reshape(-1,rgbaLoad.shape[-1])
                xyzLoad = xyzLoad.reshape(-1, xyzLoad.shape[-1])

                # remove invalid points
                rgbaLoad = rgbaLoad[~np.isnan(xyzLoad).any(axis=1)]
                xyzLoad = xyzLoad[~np.isnan(xyzLoad).any(axis=1)]

                # remove the 'a' row from the rgba data
                rgbLoad =  np.delete(rgbaLoad, (3), axis=1)

                self.xyzData = xyzLoad
                self.rgbData = rgbLoad

                if(ret == True):
                    if(self.amount == 1):
                        return self.xyzData, self.rgbData
                    else:
                        print("Amount must be set to 1 to return the capture. Exiting...")
                        exit()

                    """  
                    TODO: EXPAND FILE SAVING PROCESS!
                    """
                else:
                    np.savez("Zivid/Scans/" + name + str(i) + ".npz", xyz = self.xyzData, rgb = self.rgbData)

            i += 1 

    def getAmount(self):
        """ Returns the amount of scans in a batch """
        return self.amount

class pcd:
    """ 
    class to import data directly from Zivid scans or from .npz files with xyz and rgb data
    """

    failed = False

    def __init__(self, toScan:bool = False, name:str = None, log:bool = False):
        if(toScan == True):
            # check if a quick temporary scan is to be made 
            temp = scan()
            self.xyz, self.rgb = temp.capture(ret=True, log=log)
            del temp
        else:
            # imports xyz and rgb data from a .npz file. 
            try:
                loaded = np.load("Zivid/Scans/" + name + ".npz")
            except:
                print("File path incorrect. Exiting.")
                exit()
            
            # try and load the xyz data. If it fails it exits the program
            try:
                self.xyz = loaded["xyz"]
            except:
                print("Failed to load xyz data from the given file. Exiting.")
                exit()
            
            # try and load the rgb data. If it fails it exits the program
            try:
                self.rgb = loaded["rgb"]
            except:
                print("Failed to load rgb data from the given file. Exiting.")
                exit()

        # tries to add the xyz data to the point cloud, otherwise exits the program
        try:
            if(log == True):
                print("loading xyz data")

            xyzCopy = np.vstack([self.xyz, [0,0,0]])
            self.pointCloud = o3d.geometry.PointCloud()
            self.pointCloud.points = o3d.utility.Vector3dVector(xyzCopy)
        except:
            print("xyz data for point cloud is wrong.")
            exit()

        # checks if the given rgb data matches the length of the xyz data, otherwise exits the program
        if(len(self.rgb) == len(self.xyz)):
            try:
                if(log == True):
                    print("loading rgb data")

                rgbCopy = np.vstack([self.rgb, [0,0,0]])
                self.pointCloud.colors = o3d.utility.Vector3dVector(rgbCopy.astype(float) / 255.0)
            except:
                print("color data for point cloud is wrong.")
                exit()

    def showPointCloud(self, var, coordFrame:bool = False, log:bool = False):
        """ Shows the given pointcloud. 
        Available pointclouds: 
        - pointCloud
        - edges
        - flat
        - flatModified
        - leadingEdge """

        if(hasattr(self,var)):
            # checks if the given pointcloud exists
            pcdref = getattr(self, var)
        else:
            exit()

        # create a viewer and copy the pointcloud
        viewer = o3d.visualization.Visualizer()
        pcdCopy = o3d.geometry.PointCloud(pcdref)

        # rotate the pointcloud such that it is correctly alaigned 
        pcdCopy.rotate(pcdCopy.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))

        # aadd the pointcloud to the view window
        viewer.create_window()
        viewer.add_geometry(pcdCopy)

        # add a coordinate frame if needed
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = coordFrame

        # show the view window
        viewer.run()
        viewer.destroy_window()

    def largestCluster(self, pcdref, eps:float, minPoints:int, log:bool = False):
        """ Returns the largest cluster in the given pointcloud using the given parameters. """
        
        # Copy the pointcloud as to not make changes to the original
        pcdCopy = o3d.geometry.PointCloud(pcdref) 

        # using a DBscan, find the clusters and select the largest cluster
        labels = np.array(pcdCopy.cluster_dbscan(eps=eps,min_points=minPoints,print_progress=log))
        clusterCount = Counter(labels)
        largestClusterLabel = clusterCount.most_common(1)[0][0]
        largestClusterIndices = np.where(labels == largestClusterLabel)[0]
        pcdCopy = pcdCopy.select_by_index(largestClusterIndices)

        return pcdCopy
    
    def maskPointCloud(self, pcdref, color:np.array, inverted:bool):
        """ Returns a mask of a given pointcloud. The mask can either be inverted or not. The color to be masked needs to be a numpy array with elements between 0 and 1. """
        
        # Copy the pointcloud as to not make changes to the original
        pcdCopy = o3d.geometry.PointCloud(pcdref)

        pcdCopyPoints = np.asarray(pcdCopy.points)
        pcdCopyColors = np.asarray(pcdCopy.colors)
        
        mask = np.all(pcdCopyColors == color, axis=1)

        # check if the mask needs to be inverted or not
        if(inverted == True):
            pcdMaskPoints = pcdCopyPoints[~mask]
            pcdMaskColors = pcdCopyColors[~mask]
        else:
            pcdMaskPoints = pcdCopyPoints[mask]
            pcdMaskColors = pcdCopyColors[mask]

        # create a new pointcloud for the masked points
        maskedPcd = o3d.geometry.PointCloud()
        maskedPcd.points = o3d.utility.Vector3dVector(pcdMaskPoints)
        maskedPcd.colors = o3d.utility.Vector3dVector(pcdMaskColors)

        return maskedPcd

    def findLeadingEdge(self, eps1:float, minPoint1:int, rad:float, maxNN:int, thresh:int, eps2:float, minPoint2:int, eps3:float, minPoint3:int, dist:int = 0, rdpEps:int = 100, log:bool = False, result:bool = False):  
        if(log == True):
            self.showPointCloud("pointCloud")

        # get the largest cluster of the initial point cloud
        self.pointCloud = self.largestCluster(self.pointCloud, eps=eps1,minPoints=minPoint1, log=log)

        if(log == True):
            self.showPointCloud("pointCloud")

        # estimate the normals of the largest cluster to find its edges
        self.pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=maxNN))
        kdtree = o3d.geometry.KDTreeFlann(self.pointCloud)

        normals = np.asarray(self.pointCloud.normals)
        curvature = np.zeros(normals.shape[0])

        # using a KDTree, the earby neighbours are checked to see if there is an edge
        for i in tqdm(range(len(normals)), desc="Calculating edges",disable=not log):

            [_, idx, _] = kdtree.search_knn_vector_3d(self.pointCloud.points[i], maxNN)
            avgNormal = np.mean(normals[idx], axis=0)
            curvature[i] = np.linalg.norm(normals[i] - avgNormal)

        # only the edges within the given threshold are saved
        threshold = np.percentile(curvature, thresh)
        edgeIndices = np.where(curvature > threshold)[0]

        # the saved edges are painted red 
        self.edges = self.pointCloud.select_by_index(edgeIndices) 
        self.edges.paint_uniform_color([1,0,0])

        if(log == True):
            self.showPointCloud("edges")

        # the largest edge cluster is found to reduce the points in the point cloud
        self.edges = self.largestCluster(self.edges, eps=eps2, minPoints=minPoint2, log=log)

        if(log == True):
            self.showPointCloud("edges")

        # every point will be set a given distance as to make later calculations possible
        pcdArray = np.asarray(self.edges.points)
        for i in tqdm(range(len(pcdArray)), desc="Setting fixed distance", disable=not log):
            pcdArray[i][2] = dist

        self.flat = o3d.geometry.PointCloud()
        self.flat.points = o3d.utility.Vector3dVector(pcdArray)
        self.flat.paint_uniform_color([1,0,0])

        if(log == True):
            self.showPointCloud("flat")

        # copy the original point cloud to an array and remove its distance element to make calculations easier
        pcdXY = np.asarray(self.pointCloud.points)
        pcdXY = pcdXY[:,:2]

        # copy the largest part of the flattened edge to an array and remove its distance element to make calculations easier
        self.flat = self.largestCluster(self.flat, eps=eps3, minPoints=minPoint3, log=log)
        pcdFlatArray = np.asarray(self.flat.points)
        pcdFlatXY = pcdFlatArray[:,:2]

        if(log == True):
            self.showPointCloud("flat")

        # using the Ramer-Douglas-Pecker algorithm to find the corner of the edge
        simplifiedEdge = rdp(pcdFlatArray, epsilon=rdpEps)
        simplifiedXY = simplifiedEdge[:,:2]

        # copy all points that were found using the RDP alogrithm from the original array and the flatteded array
        indicesPcd = np.array([np.where(np.all(pcdXY == point, axis=1))[0][0] for point in simplifiedXY])
        indicesFlat = np.array([np.where(np.all(pcdFlatXY == point, axis=1))[0][0] for point in simplifiedXY])

        # delete the first and last point that were found since those can never be corners when using the RDP algorithm
        indicesPcd = np.delete(indicesPcd, len(indicesPcd)-1) 
        indicesPcd = np.delete(indicesPcd, 0)
        kdtreePcd = o3d.geometry.KDTreeFlann(self.pointCloud)

        indicesFlat = np.delete(indicesFlat, len(indicesFlat)-1)
        indicesFlat = np.delete(indicesFlat, 0)
        kdtreeFlat = o3d.geometry.KDTreeFlann(self.flat)

        # for every other point, mark it green
        for i in range(len(indicesPcd)):
            indexPcd = indicesPcd[i]
            indexFlat = indicesFlat[i]

            pointPcd = self.pointCloud.points[indexPcd]
            pointFlat = self.flat.points[indexFlat]

            [_, idPcd, _] = kdtreePcd.search_radius_vector_3d(pointPcd, 1.5)
            [_, idFlat, _] = kdtreeFlat.search_radius_vector_3d(pointFlat, 1.5)
            
            colorsPcd = np.asarray(self.pointCloud.colors)
            colorsPcd[idPcd] = [0,0,1]            
            colorsFlat = np.asarray(self.flat.colors)
            colorsFlat[idFlat] = [0,1,0]

            self.pointCloud.colors = o3d.utility.Vector3dVector(colorsPcd)
            self.flat.colors = o3d.utility.Vector3dVector(colorsFlat)

        # if more than 1 corner was found, something went wrong
        if(len(indicesPcd) != 1):
            self.failed = True
            print("failed")
        else:
            if(log == True):
                self.showPointCloud("flat")

            # if only 1 corner was found, we remove those points from the point cloud
            flatPoints = np.asarray(self.flat.points)
            flatColors = np.asarray(self.flat.colors)
            mask = np.all(flatColors == [0.0, 1.0, 0.0], axis=1)
            maskPoints = flatPoints[mask]

            self.flatModified = self.maskPointCloud(self.flat, ([0,1,0]), inverted=True)

            if(log == True):
                self.showPointCloud("flatModified")

            # the largest remaining segment is found, which sould be the leading edge
            self.flatModified = self.largestCluster(self.flatModified, eps=eps3, minPoints=minPoint3, log=log)

            if(log == True):
                self.showPointCloud("flatModified")

            # the largest remaining edge and the corner are joined together to form the entire leading edge
            flatModifiedPoints = np.asarray(self.flatModified.points)
            flatModifiedPoints = np.append(flatModifiedPoints, maskPoints, axis=0)
            self.flatModified.points = o3d.utility.Vector3dVector(flatModifiedPoints)
            self.flatModified.paint_uniform_color([0,0,1])

            if(log == True):
                self.showPointCloud("flatModified")

            # convert the flattened leading edge to just (x,y) points and then paint the points in the original point cloud with the same (x,y) points green
            flatModifiedPoints = flatModifiedPoints[:,:2]
            indicesPcd = np.array([np.where(np.all(pcdXY == point, axis=1))[0][0] for point in tqdm(flatModifiedPoints, desc="Finding leading edge", disable= not log)])

            for i in indicesPcd:
                self.pointCloud.colors[i] = [0,1,0]

            # mask the original point cloud such that just the leading edge remains
            self.leadingEdge = self.maskPointCloud(self.pointCloud, ([0,1,0]), inverted=False)

            if(log == True):
                self.showPointCloud("leadingEdge")

        # show the final found leading edge and ask if it is the correct one
        self.showPointCloud("pointCloud")

        response = input("Is the leading edge correct? (y/n)\n")

        if (response == "y"):
            self.failed = False
        else:
            self.failed = True

    def getCoordinates(self, numPoints:int = 10, log:bool = False):
        """ Returns a coordinate trace for the found leading edge """

        coords = np.array([])

        leadingEdgePcd = o3d.geometry.PointCloud(self.leadingEdge)
        self.maskedLeadingEdge = self.maskPointCloud(leadingEdgePcd, ([0,1,0]), inverted=False)
        self.showPointCloud("maskedLeadingEdge")

        maskedLeadingEdgePoints = np.asarray(self.maskedLeadingEdge.points)

        # compute pairwise distances to find the darthest points
        distanceMaxtrix = distance_matrix(maskedLeadingEdgePoints,maskedLeadingEdgePoints)          
        i,j = np.unravel_index(np.argmax(distanceMaxtrix), distanceMaxtrix.shape)
        point1, point2 = maskedLeadingEdgePoints[i], maskedLeadingEdgePoints[j]

        # interpolate points between the two farthest points
        values = np.linspace(0, 1, numPoints)
        interpolatedPoints = np.outer(1 - values, point1) + np.outer(values, point2)

        # find the closest actual points in the point cloud to these interpolated points
        kdtree = KDTree(maskedLeadingEdgePoints)
        _, indices = kdtree.query(interpolatedPoints)  # Find nearest points in the cloud

        # color the farthest points and intermediate points
        colors = np.asarray(self.maskedLeadingEdge.colors)
        if colors.shape[0] == 0:  # Initialize colors if not present
            colors = np.ones_like(maskedLeadingEdgePoints)  # White by default

        # Color the farthest points in blue
        colors[i] = [0, 0, 1]  # Blue
        colors[j] = [0, 0, 1]  # Blue

        coords = np.zeros_like(maskedLeadingEdgePoints[indices])

        # Color the interpolated (marker) points in red
        for i in range(len(indices)):
            index  = indices[i]            
            colors[index] = [1, 0, 0]  # Red

            coords[i] = maskedLeadingEdgePoints[index]

        np.set_printoptions(suppress=True)

        coords = np.round(coords, 6)
        print(coords)

        # Update the point cloud colors and visualize
        self.maskedLeadingEdge.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.maskedLeadingEdge])

        return coords

    def getFailstate(self):
        """ Returns the failstate of the leading edge detection algorithm """
        return self.failed

def main():
    # variables for the file
    scanName = "test0-"           # the name of the .npz-file that will be saved and loaded
    numberOfScans = 3               # number of scans in a batch

    initialClusterEpsilon = 1.3     # epsilon value for the inital clustering 
    InitialClusterMinPoints = 10    # minimum points for the initial clusering

    largestEdgeEpsilon = 1.5        # epsilon value for the finding of the largest edge
    largestEdgeMinPoints = 10       # minimum points for the finding of the largest edge

    edgeRefinementEpsilon = 0.70    # epsilon value for the 
    edgeRefinementMinPoints = 5     # minimum points for the

    normalsSearchRadius = 5.0       # radius for the KDTree used when caluclating the normals
    normalsMaxNeighbours = 50       # the maximum nearby neighbours for the KDTree used when calculating the normals

    edgeThreshold = 99              # the threshold to filter out irrelevant edges

    setDistace = 700                # the distance all point will be set to when flattening the point cloud

    rdpEpsilon = 100                # the epsilon used in the RDP algorithm

    coordinateTracePoints = 500     # the number of points in the leading edge coordinate trace
 
    log = True                     # to show all steps

    # setup
    if(True):
        begin = setup()
        begin.connectCamera(log=log)
        begin.setSettings(log=log)
        settings = begin.getSettings()
        cam = begin.getCamera()

    # create scans   
    if(True):
        scan1 = scan(amount=numberOfScans)
        scan1.capture(camera=cam, settings=settings, name=scanName, log=log)

    # test scans
    failed = True
    i = 0
    while(i < numberOfScans and failed == True):
        print(scanName + str(i))
        pcdTest = pcd(name=scanName + str(i))

        pcdTest.findLeadingEdge(eps1=        initialClusterEpsilon,
                                minPoint1=   InitialClusterMinPoints,
                                rad=         normalsSearchRadius,
                                maxNN=       normalsMaxNeighbours,
                                thresh=      edgeThreshold,
                                eps2=        largestEdgeEpsilon,
                                minPoint2=   largestEdgeMinPoints,
                                dist=        setDistace,
                                eps3=        edgeRefinementEpsilon,
                                minPoint3=   edgeRefinementMinPoints,
                                rdpEps=      rdpEpsilon,
                                log=         log)
        
        failed = pcdTest.getFailstate()
        i += 1
            
    if(failed == False):
        pcdTest.getCoordinates(numPoints=coordinateTracePoints)


if __name__ == "__main__":
    main()