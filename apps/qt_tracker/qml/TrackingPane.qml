import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

GroupBox {
    id: pane

    required property var controller

    title: qsTr("Tracking")

    GridLayout {
        anchors.fill: parent
        columns: 2
        columnSpacing: 8
        rowSpacing: 6

        Label { text: qsTr("Tracker") }
        ComboBox {
            Layout.fillWidth: true
            model: pane.controller.trackerNames
            currentIndex: model.indexOf(pane.controller.trackerName)
            onActivated: pane.controller.setTracker(currentText)
        }

        Label { text: qsTr("Candidates") }
        ComboBox {
            id: sourceBox
            Layout.fillWidth: true
            model: pane.controller.candidateSources
            currentIndex: model.indexOf(pane.controller.candidateSource)
            onActivated: pane.controller.setCandidateSource(currentText, pane.controller.detectorModel)
            ToolTip.visible: hovered
            ToolTip.text: pane.controller.detectorAvailable
                          ? qsTr("Template matching tracks any object; the detector tracks its known classes")
                          : qsTr("No YOLO backend found (Ultralytics or inference-models)")
        }

        Label {
            text: qsTr("Model")
            visible: modelRow.visible
        }
        RowLayout {
            id: modelRow
            Layout.fillWidth: true
            visible: pane.controller.candidateSource === "YOLO detector"
            spacing: 4

            ComboBox {
                id: modelBox
                Layout.fillWidth: true
                model: pane.controller.detectorModels
                currentIndex: model.indexOf(pane.controller.detectorModel)
                displayText: pane.controller.detectorModel.split(/[\\/]/).pop()
                onActivated: pane.controller.setCandidateSource(sourceBox.currentText, currentText)
                ToolTip.visible: hovered
                ToolTip.text: pane.controller.detectorModel
            }
            Button {
                text: qsTr("…")
                Layout.preferredWidth: 32
                onClicked: weightsDialog.open()
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Choose YOLO weights file")
            }
        }

        CheckBox {
            Layout.columnSpan: 2
            text: qsTr("McByte SAM + Cutie masks")
            visible: pane.controller.trackerName === "McByte"
            enabled: pane.controller.masksAvailable
            checked: pane.controller.useMasks
            onToggled: pane.controller.setUseMasks(checked)
            ToolTip.visible: hovered && !pane.controller.masksAvailable
            ToolTip.text: qsTr("Install the trackers[mask] extra to enable")
        }

        Label { text: qsTr("Gate width") }
        SpinBox {
            id: gateWidth
            Layout.fillWidth: true
            from: 8
            to: 1024
            editable: true
            value: pane.controller.gateWidth
            onValueModified: pane.controller.setGateSize(value, gateHeight.value)
        }

        Label { text: qsTr("Gate height") }
        SpinBox {
            id: gateHeight
            Layout.fillWidth: true
            from: 8
            to: 1024
            editable: true
            value: pane.controller.gateHeight
            onValueModified: pane.controller.setGateSize(gateWidth.value, value)
        }

        CheckBox {
            Layout.columnSpan: 2
            text: qsTr("Show candidate boxes")
            checked: pane.controller.showCandidates
            onToggled: pane.controller.setShowCandidates(checked)
        }

        Button {
            Layout.columnSpan: 2
            Layout.fillWidth: true
            text: qsTr("Stop Tracking")
            enabled: pane.controller.targetId >= 0
            onClicked: pane.controller.stopTracking()
        }
    }

    FileDialog {
        id: weightsDialog
        title: qsTr("Choose YOLO weights")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("YOLO weights (*.pt *.onnx *.engine)"), qsTr("All files (*)")]
        onAccepted: pane.controller.setCandidateSource("YOLO detector", selectedFile.toString())
    }
}
