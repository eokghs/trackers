import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

GroupBox {
    id: pane

    required property var controller

    title: qsTr("Video feed")

    ColumnLayout {
        anchors.fill: parent
        spacing: 6

        RadioButton {
            id: liveRadio
            text: qsTr("Live feed (webcam)")
            checked: true
        }
        ComboBox {
            id: cameraBox
            Layout.fillWidth: true
            enabled: liveRadio.checked
            model: pane.controller.cameras
        }
        RadioButton {
            id: fileRadio
            text: qsTr("Media file")
        }
        RowLayout {
            Layout.fillWidth: true
            enabled: fileRadio.checked

            TextField {
                id: fileField
                Layout.fillWidth: true
                readOnly: true
                placeholderText: qsTr("No file selected")
            }
            Button {
                text: qsTr("Browse…")
                onClicked: openDialog.open()
            }
        }
        RowLayout {
            Layout.fillWidth: true

            Button {
                Layout.fillWidth: true
                text: qsTr("Start")
                highlighted: true
                enabled: liveRadio.checked || openDialog.selectedFile.toString() !== ""
                onClicked: {
                    if (liveRadio.checked)
                        pane.controller.openCamera(cameraBox.currentIndex)
                    else
                        pane.controller.openFile(openDialog.selectedFile.toString())
                }
            }
            Button {
                Layout.fillWidth: true
                text: pane.controller.paused ? qsTr("Resume") : qsTr("Pause")
                enabled: pane.controller.running
                onClicked: pane.controller.togglePause()
            }
            Button {
                Layout.fillWidth: true
                text: qsTr("Close")
                enabled: pane.controller.running
                onClicked: pane.controller.closeSource()
            }
        }
    }

    FileDialog {
        id: openDialog
        title: qsTr("Open media file")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm)"), qsTr("All files (*)")]
        onAccepted: {
            fileField.text = decodeURIComponent(selectedFile.toString().replace(/^file:\/{2,3}/, ""))
            fileRadio.checked = true
        }
    }
}
