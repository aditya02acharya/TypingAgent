import React, { useState } from "react";
import { Form, Input, Button } from "antd";

import PropTypes from "prop-types";
import emitter from "./events"; //引入创建的events.js文件

import { Component } from "react";

class CustomizedForm extends Component {
  render() {
    const { fields, onChange } = this.context; // 获取context的值

    // console.log(onChange);
    return (
      <Form
        name="global_state"
        layout="inline"
        fields={fields}
        onFieldsChange={(changedFields, allFields) => {
          onChange(allFields);
        }}
      >
        <Form.Item
          name="username"
          label="Username"
          rules={[
            {
              required: true,
              message: "Username is required!",
            },
          ]}
        >
          <Input />
        </Form.Item>
      </Form>
    );
  }
}

CustomizedForm.contextTypes = {
  fields: PropTypes.array,
  onChange: PropTypes.func,
};

class Show extends Component {
  // null?
  constructor(props) {
    super(props);
    this.state = {
      fields: [
        // {
        //   name: "username",
        //   value: "Ant Design",
        // },
      ],
    };
  }

  // 创造监听事件
  componentDidMount() {
    this.eventEmitter = emitter.addListener("callMe", (data) => {
      this.setState({
        fields: data,
      });
    });
  }

  // 销毁监听事件
  componentWillUnmount() {
    emitter.removeListener("callMe", (data) => {
      this.setState({
        data,
      });
    });
  }

  render() {
    const { fields } = this.state;
    console.log(fields);
    return (
      <div>
        <pre className="language-bash">{JSON.stringify(fields, null, 2)}</pre>
        <div>hello, {JSON.stringify(fields, null, 2)}</div>
      </div>
    );
  }
}

class Reg extends Component {
  state = {
    fields: [
      {
        name: "username",
        value: "Ant Design",
      },
    ],
  };

  setFields = (newFields) => {
    this.setState({
      fields: newFields,
    });
  };

  // 给定义context赋值
  getChildContext = () => {
    return {
      fields: this.state.fields,
      onChange: this.setFields,
    };
  };

  handleClick = () => {
    emitter.emit("callMe", this.state.fields);
    // console.log(this.state.fields);
  };

  render() {
    // console.log(this.state.fields);

    return (
      <>
        <CustomizedForm />
        <pre className="language-bash">
          {JSON.stringify(this.state.fields, null, 2)}
        </pre>
        <Button htmlType="button" onClick={this.handleClick}>
          Reset
        </Button>
      </>
    );
  }
}

Reg.childContextTypes = {
  fields: PropTypes.array,
  onChange: PropTypes.func,
};

class Demo extends Component {
  render() {
    return (
      <div>
        <Reg />
        <Show />
      </div>
    );
  }
}

export default Demo;
