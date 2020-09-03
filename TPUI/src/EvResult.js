import React from "react";
import { Component } from "react";
import { Button } from "antd";
import "./css/Ev/EvResult.less";

import { EStep3, EStep4 } from "./comp/CSteps";
import { ECollapse, ECollapseAct } from "./comp/CCollapse";

import { EvMR } from "./cont/EvMR";
import { EvSR } from "./cont/EvSR";
import { EvTR, EvTR1 } from "./cont/EvTR";

class EvResult extends Component {
  render() {
    return (
      <div>
        <EStep4 />
        <ECollapseAct header="Main Result">
          <EvMR />
        </ECollapseAct>
        <ECollapseAct header="Sentence-level Results">
          <EvSR />
        </ECollapseAct>
        <ECollapseAct header="Trial-level Results">
          <EvTR />
        </ECollapseAct>
        <div className="start-ev-btn">
          <Button
            className="button"
            type="primary"
            size="large"
            onClick={this.handleClick}
          >
            Save the Model
          </Button>
        </div>
      </div>
    );
  }
}

export default EvResult;
