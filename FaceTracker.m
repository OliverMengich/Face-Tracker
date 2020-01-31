

faceDetector = vision.CascadeObjectDetector();

pointTracker = vision.PointTracker('MaxBidirectionalError',2);

cam = webcam();

videoFrame = snapshot(cam);
frameSize = size(videoFrame);

videoPlayer = vision.VideoPlayer('Position',[100 100 [frameSize(2),frameSize(1)]+30]);

%% Detection and Tracking
runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop 
    
   %get next frame
   videoFrame = snapshot(cam);
   videoFrameGray = rgb2gray(videoFrame);
   frameCount = frameCount+1;
   
   if numPts < 10
       % detection mode
       bbox = faceDetector.step(videoFrameGray);
       
       if ~isempty(bbox)
           points = detectMinEigenFeatures(videoFrameGray,'ROI',bbox(1,:));
           xyPoints = points.Location;
           numPts = size(xyPoints,1);
           release(pointTracker);
           initialize(pointTracker,xyPoints,videoFrameGray);
           
           % save a copy of the points
           oldPoints = xyPoints;
           
           bboxPoints = bbox2points(bbox(1,:));
           
           bboxPolygon = reshape(bboxPoints',1,[]);
           
           videoFrame = insertShape(videoFrame,'Polygon',bboxPolygon,'LineWidth',3);
           
           videoFrame = insertMarker(videoFrame,xyPoints,'+','Color','white');
       end 
       else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display tracked points.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
           
           
    
    
    
