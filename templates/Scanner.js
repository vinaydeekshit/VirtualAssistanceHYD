import * as ZXing from "@zxing/library";
import ScanCanvasQR from "react-pdf-image-qr-scanner";
import React, { useState, useContext, useEffect, useRef } from "react";
import { Context } from "../../Context";

const Scanner = (props) => {
  const videoRef = useRef(null);
  const canvasScannerRef = useRef();

  const { loaderStart, loaderStop, nextEnabler } = props;
  const { paxData, setPaxData } = useContext(Context);
  const { navigation, setNavigation } = useContext(Context);
  const [data, setData] = useState(null);
  const [scanFlag, setScanFlag] = useState(true);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [stopStream, setStopStream] = useState(false);
  const [fileUpload, setFileUpload] = useState(false);
  const [resultText, setResultText] = useState("");
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);

  // Function to reset the codereader I don't want use this function in useEffet as current work might get effected.
  function resetCoderReader() {
    setStopStream(false);
    setFileUpload(false);
    const codeReader = new ZXing.BrowserMultiFormatReader();
    // try{}
    codeReader
      .listVideoInputDevices()
      .then((videoInputDevices) => {
        setSelectedDeviceId(videoInputDevices[0].deviceId);

        // Handle device selection if multiple devices are available
        if (videoInputDevices.length >= 1) {
          // Bascially we are using Back camera default
          setSelectedDeviceId(videoInputDevices[0].deviceId);
        }

        codeReader.decodeFromVideoDevice(
          selectedDeviceId,
          videoRef.current,
          (result, err) => {
            if (result) {
              handleScan(result.text);
            }
            if (err && !(err instanceof ZXing.NotFoundException)) {
              console.error(err);
            }
          }
        );
      })
      .catch((err) => {
        console.error(err);
        alert("Unable to get Camera to Scan the Boarding pass!");
      });
    return () => {
      // Stop any ongoing decoding from video
      codeReader.reset();
    };
  }

  // Onload clearing the data and starting Scanner
  useEffect(() => {
    setStopStream(true);
    setFileUpload(false);
    //  //added to reset old code reader
    // if (codeReader) {
    //   codeReader.reset();
    // }

    const codeReader = new ZXing.BrowserMultiFormatReader();
    codeReader
      .listVideoInputDevices()
      .then((videoInputDevices) => {
        setSelectedDeviceId(videoInputDevices[0].deviceId);

        // Handle device selection if multiple devices are available
        if (videoInputDevices.length >= 1) {
          setSelectedDeviceId(videoInputDevices[0].deviceId);
        }

        codeReader.decodeFromVideoDevice(
          selectedDeviceId,
          videoRef.current,
          (result, err) => {
            if (result) {
              handleScan(result.text);
              // Handle the scanned result here
            }
            if (err && !(err instanceof ZXing.NotFoundException)) {
              console.error(err);
            }
          }
        );
      })
      .catch((err) => {
        console.error(err);
      });
    return () => {
      // Stop any ongoing decoding from video
      codeReader.reset();
    };
  }, []);

  // On every Scan result
  useEffect(() => {
    if (resultText !== "No QR code found" && resultText.length > 1) {
      manualQR(resultText);
    }
  }, [resultText]);
  //  On every scan
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    const handlePermissionRevoke = () => {
      navigator.mediaDevices.getUserMedia({ video: false });
    };

    if (scanFlag) {
      setNavigation([
        [true, true],
        [true, false],
      ]);
    } else {
      setNavigation([
        [true, true],
        [true, false],
      ]);
      // Revoke camera permissions when not scanning
      handlePermissionRevoke();
    }

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [scanFlag]);

  // scanfile pdf/Image reader
  async function scanFile(selectedFile) {
    console.time()
    setResultText("");
    try {
      loaderStart();
      const qrCode = await canvasScannerRef.current.scanFile(selectedFile);
      if (qrCode == null) {
        loaderStop();
        alert("No Qr Detected!");
      } else {
        loaderStop();
        setResultText(qrCode);
      }
    } catch (e) {
      loaderStop();
      console.log(e?.name);
      if (e?.name === "InvalidPDFException") {
        setResultText("");
      } else if (e instanceof Event) {
        setResultText("");
      } else if (e?.name === "TypeError") {
        console.log(e);
        alert("No file selected");
        setResultText("");
      } else {
        console.log(e);
        setResultText("");
      }
    }
    console.timeEnd()
  }

  // Bulian date to normal date converter
  function isCurrentDate(rawdate) {
    const julianDate = rawdate;
    const currentYear = new Date().getFullYear();
    const startDate = new Date(currentYear, 0, 1);
    const bpDate = new Date(startDate);
    bpDate.setDate(startDate.getDate() + parseInt(julianDate) - 1);

    // Get current date
    const currentDate = new Date();
    console.log(
      bpDate.getDate(),
      currentDate.getDate(),
      bpDate.getMonth(),
      currentDate.getMonth(),
      bpDate.getFullYear(),
      currentDate.getFullYear()
    );

    // Compare the generated date with the current date
    return (
      bpDate.getDate() === currentDate.getDate() &&
      bpDate.getMonth() === currentDate.getMonth() &&
      bpDate.getFullYear() === currentDate.getFullYear()
    );
  }

  // Main Logic, Extracting data from M1 String
  function manualQR(data) {
    const mannualData = data;
    let PAX = "";
    let FLNO = "";
    let ORIGIN = "";
    let DEST = "";
    let PNR = "";
    let SEATNO = "";
    let bpdate = "";
    let check = "";
    try {
      check = mannualData.substring(0, 2).trim();
      bpdate = isCurrentDate(mannualData.substring(44, 47));
      if (check !== "M1") {
        throw "Invalid QR";
      }
      if (!bpdate) {
        throw "Please check your Journey Date!";
      }
      PAX = mannualData.substring(2, 22).trim();

      FLNO = mannualData.substring(36, 44).trim();

      ORIGIN = mannualData.substring(30, 33);

      DEST = mannualData.substring(33, 36);

      PNR = mannualData.substring(23, 30).trim();

      SEATNO = mannualData.substring(49, 52);
    } catch (error) {
      alert(error);
    }

    let paxDetails = {
      name: PAX,
      pnr: PNR,
      flight: FLNO,
      origin: ORIGIN,
      desti: DEST,
      seat: SEATNO,
      strings: data,
    };

    if (
      paxDetails.name.length !== 0 &&
      paxDetails.pnr.length !== 0 &&
      paxDetails.flight.length !== 0 &&
      paxDetails.desti.length !== 0 &&
      paxDetails.origin.length !== 0
    ) {
      setNavigation([
        [false, true],
        [true, true],
      ]);
      setPaxData(paxDetails);
      setScanFlag(false);
      nextEnabler();
      setStopStream(true);
      try {
        const mediaStream = videoRef.current.srcObject; // Get the media stream
        if (mediaStream) {
          const tracks = mediaStream.getTracks();
          tracks.forEach((track) => track.stop()); // Stop each track (video and audio)
        }
      } catch {
        console.log("No media Objects");
      }
    }
  }
  
  const handleScan = (result) => {
    if (scanFlag && result) {
      if (result.length > 0) {
        manualQR(result);
      }
    }
  };


  return (
    <div style={{ position: "relative" }}>
      <style>
        {`
        #buttonParent{
          justify-content:center;
        }
        
        `}
      </style>
      {/* Scanner */}
      {!fileUpload && (
        <>
          <video ref={videoRef} playsInline className="qrScanner" />
          <button
            className="uploadManualButton"
            onClick={() => setFileUpload(true)}
          >
            Upload Boarding pass
          </button>
        </>
      )}
      {/* file Upload */}
      {fileUpload && (
        <div className="manulPDFContainer">
          <span>
            <h2 style={{ color: "#ed1b24", textAlign: "center" }}>
              Upload the Boarding pass PDF/Image file.
            </h2>
            <h5
              style={{
                color: "#07578d",
                marginBottom: "2rem",
                textAlign: "center",
              }}
            >
              It may take few seconds. We don't store your Files
            </h5>
          </span>
          <span className="centerUploader">
            <ScanCanvasQR ref={canvasScannerRef} />
            <input
              type="file"
              onChange={(e) => {
                scanFile(e.target.files[0]);
              }}
              className="fileInput"
            />
          </span>
          <button className="scanFromManulButton" onClick={resetCoderReader}>
            Use Camera to Scan
          </button>
        </div>
      )}
    </div>
  );
};
export default Scanner;
