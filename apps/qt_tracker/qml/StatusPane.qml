import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GroupBox {
    id: pane

    required property var controller

    title: qsTr("Status")

    GridLayout {
        anchors.fill: parent
        columns: 2
        columnSpacing: 12
        rowSpacing: 4

        Label { text: qsTr("Source") }
        Label {
            Layout.fillWidth: true
            elide: Text.ElideMiddle
            text: pane.controller.running ? pane.controller.sourceLabel : qsTr("None")
        }
        Label { text: qsTr("Format") }
        Label { Layout.fillWidth: true; text: pane.controller.resolution }
        Label { text: qsTr("Target") }
        Label {
            Layout.fillWidth: true
            font.bold: true
            color: pane.controller.status === "Tracking" ? "#2e9e44"
                 : pane.controller.status === "Lost" ? "#d23c3c" : palette.text
            text: pane.controller.targetId >= 0
                  ? qsTr("%1 (ID %2)").arg(pane.controller.status).arg(pane.controller.targetId)
                  : pane.controller.status
        }
        Label { text: qsTr("Score") }
        Label { Layout.fillWidth: true; text: pane.controller.score.toFixed(2) }
        Label { text: qsTr("Speed") }
        Label { Layout.fillWidth: true; text: qsTr("%1 fps").arg(pane.controller.fps.toFixed(1)) }
        Label { text: qsTr("Frame") }
        Label {
            Layout.fillWidth: true
            text: pane.controller.frameCount > 0
                  ? qsTr("%1 / %2").arg(pane.controller.frameIndex).arg(pane.controller.frameCount)
                  : pane.controller.frameIndex
        }
    }
}
