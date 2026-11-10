import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

GroupBox {
    id: pane

    required property var controller

    title: qsTr("Processed video")

    ColumnLayout {
        anchors.fill: parent
        spacing: 6

        Button {
            Layout.fillWidth: true
            text: pane.controller.saving ? qsTr("Stop Saving") : qsTr("Save")
            enabled: pane.controller.running
            onClicked: pane.controller.saving ? pane.controller.stopSaving() : saveDialog.open()
        }
        Label {
            Layout.fillWidth: true
            visible: pane.controller.saving
            elide: Text.ElideMiddle
            text: qsTr("Recording to %1").arg(pane.controller.savePath)
        }
    }

    FileDialog {
        id: saveDialog
        title: qsTr("Save tracked video")
        fileMode: FileDialog.SaveFile
        defaultSuffix: "mp4"
        nameFilters: [qsTr("MP4 video (*.mp4)"), qsTr("AVI video (*.avi)")]
        onAccepted: pane.controller.startSaving(selectedFile.toString())
    }
}
