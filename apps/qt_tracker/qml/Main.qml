import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window

    required property var controller

    width: 1280
    height: 800
    minimumWidth: 960
    minimumHeight: 600
    visible: true
    title: qsTr("Video Object Tracker")

    RowLayout {
        anchors.fill: parent
        spacing: 0

        ScrollView {
            Layout.preferredWidth: 320
            Layout.fillHeight: true
            contentWidth: availableWidth

            ColumnLayout {
                width: parent.width
                spacing: 8

                SourcePane {
                    controller: window.controller
                    Layout.fillWidth: true
                    Layout.margins: 8
                    Layout.bottomMargin: 0
                }
                TrackingPane {
                    controller: window.controller
                    Layout.fillWidth: true
                    Layout.leftMargin: 8
                    Layout.rightMargin: 8
                }
                SavePane {
                    controller: window.controller
                    Layout.fillWidth: true
                    Layout.leftMargin: 8
                    Layout.rightMargin: 8
                }
                StatusPane {
                    controller: window.controller
                    Layout.fillWidth: true
                    Layout.leftMargin: 8
                    Layout.rightMargin: 8
                    Layout.bottomMargin: 8
                }
            }
        }

        VideoView {
            controller: window.controller
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    MessageBar {
        id: messageBar
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 16
    }

    Connections {
        target: window.controller
        function onMessage(text) { messageBar.show(text) }
    }
}
