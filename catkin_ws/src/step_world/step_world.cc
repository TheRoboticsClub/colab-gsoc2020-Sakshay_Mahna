#include <boost/bind.hpp>
#include "gazebo/common/Plugin.hh"
#include "gazebo/msgs/msgs.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/transport/transport.hh"

namespace gazebo
{
	class WorldEdit: public WorldPlugin
	{
		public: void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
		{
			// Create a new transport node
			this->node.reset(new transport::Node());
			
			// Initialize the node with the world name
			this->node->Init(_parent->Name());
			
			// Create a publisher
			this->pub = this->node->Advertise<msgs::WorldControl>("~/world_control");
			
			// Listen to the update event. Event is broadcast every simulation
			// iteration
			this->updateConnection = event::Events::ConnectWorldUpdateEnd(
			boost::bind(&WorldEdit::OnUpdate, this));
			
			// Configure the initial message to the system
			msgs::WorldControl worldControlMsg;
			
			// Set the world to unpaused
			worldControlMsg.set_pause(0);
			
			// Set the step flag to true
			worldControlMsg.set_step(1);
			
			// Publish the initial message
			this->pub->Publish(worldControlMsg);
			
		}
		
		// Called by the world update start event
		public: void OnUpdate()
		{
			// Throttle Publication
            gazebo::common::Time::MSleep(3);
			
			msgs::WorldControl msg;
			msg.set_step(1);
			this->pub->Publish(msg);
		}
		
		// Pointer to the world_controller
		private: transport::NodePtr node;
		private: transport::PublisherPtr pub;
		
		// Pointer to the update event connection
		private: event::ConnectionPtr updateConnection;
	};
	
	// Register this plugin with the simulator
	GZ_REGISTER_WORLD_PLUGIN(WorldEdit);
}
