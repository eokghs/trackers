import QtQuick
import QtQuick.Controls
import TrackerApp

Rectangle {
    id: root

    required property var controller

    readonly property real frameScale: video.hasFrame
        ? Math.min(video.width / video.frameWidth, video.height / video.frameHeight) : 1.0

    color: "#101114"

    VideoFrameItem {
        id: video
        anchors.fill: parent
        Component.onCompleted: root.controller.attachView(video)
    }

    Label {
        anchors.centerIn: parent
        visible: !video.hasFrame
        color: "#9aa0a6"
        text: qsTr("Select a video feed and press Start")
    }

    Label {
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 10
        visible: video.hasFrame && root.controller.targetId < 0
        padding: 6
        color: "white"
        text: qsTr("Left-click an object to start tracking")
        background: Rectangle { color: "#99000000"; radius: 4 }
    }

    Rectangle {
        id: gatePreview
        visible: mouseArea.containsMouse && video.hasFrame
        width: root.controller.gateWidth * root.frameScale
        height: root.controller.gateHeight * root.frameScale
        x: mouseArea.mouseX - width / 2
        y: mouseArea.mouseY - height / 2
        color: "transparent"
        border.color: "#ffd54f"
        border.width: 1
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton
        cursorShape: video.hasFrame ? Qt.CrossCursor : Qt.ArrowCursor
        onClicked: (mouse) => {
            const p = video.mapToImage(mouse.x, mouse.y)
            if (p.x >= 0)
                root.controller.selectTarget(p.x, p.y)
        }
    }
}
