import QtQuick
import QtQuick.Controls
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
            text: qsTr("Show search candidates")
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
}
