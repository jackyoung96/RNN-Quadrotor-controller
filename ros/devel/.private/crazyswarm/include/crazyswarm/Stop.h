// Generated by gencpp from file crazyswarm/Stop.msg
// DO NOT EDIT!


#ifndef CRAZYSWARM_MESSAGE_STOP_H
#define CRAZYSWARM_MESSAGE_STOP_H

#include <ros/service_traits.h>


#include <crazyswarm/StopRequest.h>
#include <crazyswarm/StopResponse.h>


namespace crazyswarm
{

struct Stop
{

typedef StopRequest Request;
typedef StopResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct Stop
} // namespace crazyswarm


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::crazyswarm::Stop > {
  static const char* value()
  {
    return "d44d7e9aa94d069ed5834dbd7329e1bb";
  }

  static const char* value(const ::crazyswarm::Stop&) { return value(); }
};

template<>
struct DataType< ::crazyswarm::Stop > {
  static const char* value()
  {
    return "crazyswarm/Stop";
  }

  static const char* value(const ::crazyswarm::Stop&) { return value(); }
};


// service_traits::MD5Sum< ::crazyswarm::StopRequest> should match
// service_traits::MD5Sum< ::crazyswarm::Stop >
template<>
struct MD5Sum< ::crazyswarm::StopRequest>
{
  static const char* value()
  {
    return MD5Sum< ::crazyswarm::Stop >::value();
  }
  static const char* value(const ::crazyswarm::StopRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::crazyswarm::StopRequest> should match
// service_traits::DataType< ::crazyswarm::Stop >
template<>
struct DataType< ::crazyswarm::StopRequest>
{
  static const char* value()
  {
    return DataType< ::crazyswarm::Stop >::value();
  }
  static const char* value(const ::crazyswarm::StopRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::crazyswarm::StopResponse> should match
// service_traits::MD5Sum< ::crazyswarm::Stop >
template<>
struct MD5Sum< ::crazyswarm::StopResponse>
{
  static const char* value()
  {
    return MD5Sum< ::crazyswarm::Stop >::value();
  }
  static const char* value(const ::crazyswarm::StopResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::crazyswarm::StopResponse> should match
// service_traits::DataType< ::crazyswarm::Stop >
template<>
struct DataType< ::crazyswarm::StopResponse>
{
  static const char* value()
  {
    return DataType< ::crazyswarm::Stop >::value();
  }
  static const char* value(const ::crazyswarm::StopResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // CRAZYSWARM_MESSAGE_STOP_H
