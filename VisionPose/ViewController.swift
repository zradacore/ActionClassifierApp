//
//  ViewController.swift
//  VisionPose
//
//  Created by Albert on 6/24/20.
//

import UIKit
import Vision

class ViewController: UIViewController {

    var imageSize = CGSize.zero

    @IBOutlet weak var previewImageView: UIImageView!
    
    private let videoCapture = VideoCapture()
    
    private var currentFrame: CGImage?
    
    var isShowCalculateResult = false
    @IBOutlet weak var resultPoints: UITextView!
    @IBOutlet weak var finalPoints: UITextView!
    
    @IBOutlet weak var actionLabel: UILabel!
    let predictor = Predictor()
    var poseObservations = [VNHumanBodyPoseObservation]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupAndBeginCapturingVideoFrames()
        
        resultPoints.isHidden = !isShowCalculateResult
        finalPoints.isHidden = !isShowCalculateResult
    }

    override func viewWillDisappear(_ animated: Bool) {
        videoCapture.stopCapturing {
            super.viewWillDisappear(animated)
        }
    }

    override func viewWillTransition(to size: CGSize,
                                     with coordinator: UIViewControllerTransitionCoordinator) {
        // Reinitilize the camera to update its output stream with the new orientation.
        setupAndBeginCapturingVideoFrames()
    }
    
    private func setupAndBeginCapturingVideoFrames() {
        videoCapture.setUpAVCapture { error in
            if let error = error {
                print("Failed to setup camera with error \(error)")
                return
            }

            self.videoCapture.delegate = self

            self.videoCapture.startCapturing()
        }
    }
    
    @IBAction func onCameraButtonTapped(_ sender: Any) {
        videoCapture.flipCamera { error in
            if let error = error {
                print("Failed to flip camera with error \(error)")
            }
        }
    }
    
    func storeObservation(_ observation: VNHumanBodyPoseObservation) {
        if poseObservations.count >= 30 {
            poseObservations.removeFirst()
        }
        poseObservations.append(observation)
    }
    
    func estimation(_ cgImage:CGImage) {
        imageSize = CGSize(width: cgImage.width, height: cgImage.height)

        let requestHandler = VNImageRequestHandler(cgImage: cgImage)

        // Create a new request to recognize a human body pose.
        let request = VNDetectHumanBodyPoseRequest(completionHandler: bodyPoseHandler)

        do {
            // Perform the body pose-detection request.
            try requestHandler.perform([request])
        } catch {
            print("Unable to perform the request: \(error).")
        }
    }
    
    func bodyPoseHandler(request: VNRequest, error: Error?) {
        guard let observations =
                request.results as? [VNRecognizedPointsObservation] else { return }
        
        // Process each observation to find the recognized body pose points.
        if observations.count == 0 {
            guard let currentFrame = self.currentFrame else {
                return
            }
            let image = UIImage(cgImage: currentFrame)
            DispatchQueue.main.async {
                self.previewImageView.image = image
            }
        } else {
            observations.forEach { processObservation($0) }
        }
        
        if let result = observations.first {
            storeObservation(result as! VNHumanBodyPoseObservation)
            labelActionType()
        }
    }
    
    func processObservation(_ observation: VNRecognizedPointsObservation) {
        
        var resultPointsStr = "result:"
        var finalPointsStr = "final:"
        
        // Retrieve all torso points.
        guard let recognizedPoints =
                try? observation.recognizedPoints(forGroupKey: VNRecognizedPointGroupKey.all) else {
            return
        }
        
        let imagePoints: [CGPoint] = recognizedPoints.values.compactMap {
            guard $0.confidence > 0 else { return nil }
            
            resultPointsStr = resultPointsStr + "\n(\(String(format: "%.6f", $0.location.x)),\(String(format: "%.6f", $0.location.y)))"
            
            let result = VNImagePointForNormalizedPoint($0.location,
                                                        Int(imageSize.width),
                                                        Int(imageSize.height))
            finalPointsStr = finalPointsStr + "\n(\(String(format: "%.4f", result.x)),\(String(format: "%.4f", result.y)))"
            return result
        }
        
//        print("pose cnt \(imagePoints.count): \(imagePoints)")
        DispatchQueue.main.async {
            self.resultPoints.text = resultPointsStr
            self.finalPoints.text = finalPointsStr
        }
        
        let image = currentFrame?.drawPoints(points: imagePoints)
        DispatchQueue.main.async {
            self.previewImageView.image = image
        }
    }

}



// MARK: - VideoCaptureDelegate

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ videoCapture: VideoCapture, didCaptureFrame capturedImage: CGImage?) {

        guard let image = capturedImage else {
            fatalError("Captured image is null")
        }

        currentFrame = image

        estimation(image)
    }
    
    func labelActionType() {
        guard let actionClassifier = try? EstimationModel(configuration: MLModelConfiguration()),
              let poseMultiArray = prepareInputWithObservations(poseObservations),
              let predictions = try? actionClassifier.prediction(poses: poseMultiArray) else {
            return
        }
        let label = predictions.label
        let confidence = predictions.labelProbabilities[label] ?? 0
//        print(confidence, label)
        let isOverThreshold = confidence > 0.5
        DispatchQueue.main.async {
            self.actionLabel.isHidden = !isOverThreshold
            self.actionLabel.text = isOverThreshold ? label : ""
        }
    }
    
    func videoCaptureBuffer(_ videoCapture: VideoCapture, didCaptureBuffer buffer: CMSampleBuffer) {
//        if let poseObservation = try? self.predictor.performBodyPoseRequest(buffer) {
//            // Fetch body joints from the observation and overlay them on the player.
////            let joints = getBodyJointsFor(observation: poseObservation)
////            DispatchQueue.main.async {
////                self.bodyView.joints = joints
////            }
//        }
//
//        guard let prediction = try? self.predictor.makePrediction() else {
//            return
//        }
//
//        let label = prediction.label
//
//        DispatchQueue.main.async {
//            self.actionLabel.text = "\(label)"
//        }
    }
    
    func prepareInputWithObservations(_ observations: [VNHumanBodyPoseObservation]) -> MLMultiArray? {
        let numAvailableFrames = observations.count
        let observationsNeeded = 30
        var multiArrayBuffer = [MLMultiArray]()

        for frameIndex in 0 ..< min(numAvailableFrames, observationsNeeded) {
            let pose = observations[frameIndex]
            do {
                let oneFrameMultiArray = try pose.keypointsMultiArray()
                multiArrayBuffer.append(oneFrameMultiArray)
            } catch {
                continue
            }
        }
        
        // If poseWindow does not have enough frames (45) yet, we need to pad 0s
        if numAvailableFrames < observationsNeeded {
            for _ in 0 ..< (observationsNeeded - numAvailableFrames) {
                do {
                    let oneFrameMultiArray = try MLMultiArray(shape: [1, 3, 18], dataType: .double)
                    try resetMultiArray(oneFrameMultiArray)
                    multiArrayBuffer.append(oneFrameMultiArray)
                } catch {
                    continue
                }
            }
        }
        return MLMultiArray(concatenating: [MLMultiArray](multiArrayBuffer), axis: 0, dataType: .float)
    }
    
    func resetMultiArray(_ predictionWindow: MLMultiArray, with value: Double = 0.0) throws {
        let pointer = try UnsafeMutableBufferPointer<Double>(predictionWindow)
        pointer.initialize(repeating: value)
    }
}
