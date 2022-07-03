// Generated by gencpp from file crazyswarm/SetGroupMaskResponse.msg
// DO NOT EDIT!


#ifndef CRAZYSWARM_MESSAGE_SETGROUPMASKRESPONSE_H
#define CRAZYSWARM_MESSAGE_SETGROUPMASKRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace crazyswarm
{
template <class ContainerAllocator>
struct SetGroupMaskResponse_
{
  typedef SetGroupMaskResponse_<ContainerAllocator> Type;

  SetGroupMaskResponse_()
    {
    }
  SetGroupMaskResponse_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> const> ConstPtr;

}; // struct SetGroupMaskResponse_

typedef ::crazyswarm::SetGroupMaskResponse_<std::allocator<void> > SetGroupMaskResponse;

typedef boost::shared_ptr< ::crazyswarm::SetGroupMaskResponse > SetGroupMaskResponsePtr;
typedef boost::shared_ptr< ::crazyswarm::SetGroupMaskResponse const> SetGroupMaskResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace crazyswarm

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "crazyswarm/SetGroupMaskResponse";
  }

  static const char* value(const ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetGroupMaskResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::crazyswarm::SetGroupMaskResponse_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // CRAZYSWARM_MESSAGE_SETGROUPMASKRESPONSE_H
